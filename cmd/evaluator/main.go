// cmd/evaluator/main.go
// LLM-as-a-Judge evaluation service.
// Pulls inference samples from Redis, scores them via OpenAI/Ollama,
// and pushes evaluation metrics back to Prometheus.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// EvalSample is stored in Redis by the ingester for async evaluation.
type EvalSample struct {
	EventID      string  `json:"event_id"`
	ModelID      string  `json:"model_id"`
	Prompt       string  `json:"prompt"`
	Response     string  `json:"response"`
	GroundTruth  string  `json:"ground_truth,omitempty"`
	TaskType     string  `json:"task_type"` // classification | qa | summarization | agent
	Timestamp    string  `json:"timestamp"`
}

// JudgeScore is the structured output expected from the LLM judge.
type JudgeScore struct {
	Score      float64 `json:"score"`       // 0.0–1.0
	Grounding  float64 `json:"grounding"`   // factual accuracy / citation quality
	Coherence  float64 `json:"coherence"`   // logical flow
	Relevance  float64 `json:"relevance"`   // on-topic
	Hallucination float64 `json:"hallucination"` // 0=clean 1=hallucinated
	Reasoning  string  `json:"reasoning"`
}

// LLMJudgeClient calls an OpenAI-compatible endpoint.
type LLMJudgeClient struct {
	endpoint string
	apiKey   string
	model    string
	client   *http.Client
}

func newJudgeClient() *LLMJudgeClient {
	return &LLMJudgeClient{
		endpoint: envOrDefault("JUDGE_ENDPOINT", "http://localhost:11434/v1/chat/completions"),
		apiKey:   envOrDefault("JUDGE_API_KEY", "ollama"),
		model:    envOrDefault("JUDGE_MODEL", "llama3.1:8b"),
		client:   &http.Client{Timeout: 60 * time.Second},
	}
}

const judgeSystemPrompt = `You are an expert ML evaluation judge.
Given a model prompt and its response, score it strictly and output ONLY valid JSON.
Scores are 0.0 (worst) to 1.0 (best). Be harsh — 0.7+ means genuinely good output.`

func judgeUserPrompt(sample EvalSample) string {
	gt := ""
	if sample.GroundTruth != "" {
		gt = fmt.Sprintf("\nGround truth: %s", sample.GroundTruth)
	}
	return fmt.Sprintf(`Task type: %s
Prompt: %s
Model response: %s%s

Output ONLY this JSON (no markdown, no explanation outside the JSON):
{
  "score": <overall 0.0-1.0>,
  "grounding": <factual accuracy 0.0-1.0>,
  "coherence": <logical flow 0.0-1.0>,
  "relevance": <on-topic 0.0-1.0>,
  "hallucination": <0=clean 1=fully hallucinated>,
  "reasoning": "<one sentence>"
}`, sample.TaskType, sample.Prompt, sample.Response, gt)
}

func (j *LLMJudgeClient) Judge(ctx context.Context, sample EvalSample) (*JudgeScore, error) {
	body := map[string]interface{}{
		"model": j.model,
		"messages": []map[string]string{
			{"role": "system", "content": judgeSystemPrompt},
			{"role": "user", "content": judgeUserPrompt(sample)},
		},
		"temperature": 0.1,
	}
	b, _ := json.Marshal(body)
	req, _ := http.NewRequestWithContext(ctx, "POST", j.endpoint, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+j.apiKey)

	resp, err := j.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("judge request: %w", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)

	var openAIResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &openAIResp); err != nil || len(openAIResp.Choices) == 0 {
		return nil, fmt.Errorf("bad judge response: %s", string(raw))
	}
	content := openAIResp.Choices[0].Message.Content

	var score JudgeScore
	if err := json.Unmarshal([]byte(content), &score); err != nil {
		return nil, fmt.Errorf("parse judge JSON: %w — raw: %s", err, content)
	}
	return &score, nil
}

func main() {
	redisAddr := envOrDefault("REDIS_ADDR", "localhost:6379")
	metricsPort := envOrDefault("EVAL_METRICS_PORT", "2113")
	evalQueue := envOrDefault("EVAL_QUEUE", "eval:queue")
	evalInterval := 5 * time.Second
	if v := os.Getenv("EVAL_INTERVAL_SEC"); v != "" {
		if sec, err := strconv.Atoi(v); err == nil {
			evalInterval = time.Duration(sec) * time.Second
		}
	}

	// ── Redis ────────────────────────────────────────────────────────────────
	rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
	ctx := context.Background()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("Redis unavailable: %v", err)
	}
	log.Printf("Evaluator connected to Redis at %s", redisAddr)

	// ── Prometheus metrics ───────────────────────────────────────────────────
	evalScore := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "llm_eval_score",
		Help:    "LLM-as-judge overall quality score (0-1) per model and task.",
		Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
	}, []string{"model_id", "task_type"})

	evalGrounding := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "llm_eval_grounding",
		Help:    "Grounding / factual accuracy score from LLM judge.",
		Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
	}, []string{"model_id"})

	evalHallucination := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "llm_eval_hallucination_rate",
		Help: "Rolling hallucination rate (0=clean, 1=hallucinated) per model.",
	}, []string{"model_id"})

	evalCoherence := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "llm_eval_coherence",
		Help:    "Response coherence score from LLM judge.",
		Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
	}, []string{"model_id"})

	evalTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "llm_eval_total",
		Help: "Total evaluations run by model.",
	}, []string{"model_id", "status"})

	evalQueueDepth := prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "eval_queue_depth",
		Help: "Number of samples waiting in the evaluation queue.",
	})

	hallucinationWindows := map[string]*rollingAvg{}

	reg := prometheus.NewRegistry()
	reg.MustRegister(evalScore, evalGrounding, evalHallucination,
		evalCoherence, evalTotal, evalQueueDepth)

	// ── Metrics HTTP ─────────────────────────────────────────────────────────
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"status":"ok","service":"evaluator"}`)
	})
	go func() {
		log.Printf("Evaluator metrics on :%s", metricsPort)
		if err := http.ListenAndServe(":"+metricsPort, mux); err != nil {
			log.Fatal(err)
		}
	}()

	judge := newJudgeClient()
	log.Printf("LLM judge: %s → model %s", judge.endpoint, judge.model)

	// ── Signal handling ───────────────────────────────────────────────────────
	evalCtx, cancel := context.WithCancel(ctx)
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("Evaluator shutting down...")
		cancel()
	}()

	ticker := time.NewTicker(evalInterval)
	defer ticker.Stop()

	log.Printf("Evaluator polling queue=%s every %s", evalQueue, evalInterval)
	for {
		select {
		case <-evalCtx.Done():
			return
		case <-ticker.C:
		}

		// Track queue depth
		if depth, err := rdb.LLen(evalCtx, evalQueue).Result(); err == nil {
			evalQueueDepth.Set(float64(depth))
		}

		// Pop up to 5 samples per tick
		for i := 0; i < 5; i++ {
			raw, err := rdb.LPop(evalCtx, evalQueue).Result()
			if err == redis.Nil {
				break
			}
			if err != nil {
				log.Printf("Redis pop error: %v", err)
				break
			}

			var sample EvalSample
			if err := json.Unmarshal([]byte(raw), &sample); err != nil {
				log.Printf("Parse sample error: %v", err)
				evalTotal.WithLabelValues("unknown", "parse_error").Inc()
				continue
			}

			score, err := judge.Judge(evalCtx, sample)
			if err != nil {
				log.Printf("Judge error event=%s: %v", sample.EventID, err)
				evalTotal.WithLabelValues(sample.ModelID, "judge_error").Inc()
				continue
			}

			evalScore.WithLabelValues(sample.ModelID, sample.TaskType).Observe(score.Score)
			evalGrounding.WithLabelValues(sample.ModelID).Observe(score.Grounding)
			evalCoherence.WithLabelValues(sample.ModelID).Observe(score.Coherence)
			evalTotal.WithLabelValues(sample.ModelID, "success").Inc()

			if _, ok := hallucinationWindows[sample.ModelID]; !ok {
				hallucinationWindows[sample.ModelID] = &rollingAvg{size: 50}
			}
			hallucinationWindows[sample.ModelID].add(score.Hallucination)
			evalHallucination.WithLabelValues(sample.ModelID).Set(
				hallucinationWindows[sample.ModelID].avg())

			log.Printf("Eval event=%s model=%s score=%.2f grounding=%.2f hallucination=%.2f | %s",
				sample.EventID, sample.ModelID, score.Score, score.Grounding,
				score.Hallucination, score.Reasoning)
		}
	}
}

type rollingAvg struct {
	vals []float64
	size int
}

func (r *rollingAvg) add(v float64) {
	if len(r.vals) >= r.size {
		r.vals = r.vals[1:]
	}
	r.vals = append(r.vals, v)
}

func (r *rollingAvg) avg() float64 {
	if len(r.vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range r.vals {
		sum += v
	}
	return sum / float64(len(r.vals))
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
