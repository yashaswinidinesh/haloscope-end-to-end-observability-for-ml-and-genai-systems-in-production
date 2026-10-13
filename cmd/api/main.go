// cmd/api/main.go
// REST API gateway — receives inference events from models, pushes to Kafka,
// queues LLM eval samples in Redis. Also exposes /dashboard summary endpoint.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/confluentinc/confluent-kafka-go/v2/kafka"
	"github.com/gin-gonic/gin"
	"github.com/go-redis/redis/v8"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

type InferencePayload struct {
	ModelID      string             `json:"model_id" binding:"required"`
	ModelVersion string             `json:"model_version" binding:"required"`
	Latency      float64            `json:"latency_ms" binding:"required"`
	Features     map[string]float64 `json:"input_features"`
	Prediction   float64            `json:"prediction"`
	Confidence   float64            `json:"confidence"`
	ActualLabel  *float64           `json:"actual_label,omitempty"`
	ErrorType    string             `json:"error_type,omitempty"`
	Tags         map[string]string  `json:"tags"`
	// LLM eval fields (optional — only for GenAI models)
	Prompt      string `json:"prompt,omitempty"`
	Response    string `json:"response,omitempty"`
	GroundTruth string `json:"ground_truth,omitempty"`
	TaskType    string `json:"task_type,omitempty"`
}

func main() {
	kafkaBroker := envOrDefault("KAFKA_BROKER", "localhost:9092")
	kafkaTopic := envOrDefault("KAFKA_TOPIC", "ml-inference-events")
	redisAddr := envOrDefault("REDIS_ADDR", "localhost:6379")
	apiPort := envOrDefault("API_PORT", "8080")
	metricsPort := envOrDefault("API_METRICS_PORT", "2114")

	// ── Kafka producer ───────────────────────────────────────────────────────
	producer, err := kafka.NewProducer(&kafka.ConfigMap{
		"bootstrap.servers": kafkaBroker,
		"acks":              "all",
	})
	if err != nil {
		log.Fatalf("Kafka producer init: %v", err)
	}
	defer producer.Close()
	go func() {
		for e := range producer.Events() {
			if msg, ok := e.(*kafka.Message); ok && msg.TopicPartition.Error != nil {
				log.Printf("Kafka delivery error: %v", msg.TopicPartition.Error)
			}
		}
	}()

	// ── Redis ────────────────────────────────────────────────────────────────
	rdb := redis.NewClient(&redis.Options{Addr: redisAddr})
	ctx := context.Background()
	if err := rdb.Ping(ctx).Err(); err != nil {
		log.Fatalf("Redis unavailable: %v", err)
	}

	// ── Prometheus ────────────────────────────────────────────────────────────
	apiRequests := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "api_requests_total",
		Help: "Total API requests by endpoint and status.",
	}, []string{"endpoint", "status"})
	apiLatency := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "api_request_latency_ms",
		Help:    "API request latency.",
		Buckets: []float64{5, 10, 25, 50, 100, 250, 500},
	}, []string{"endpoint"})

	reg := prometheus.NewRegistry()
	reg.MustRegister(apiRequests, apiLatency)
	go func() {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
		log.Printf("API metrics on :%s", metricsPort)
		http.ListenAndServe(":"+metricsPort, mux)
	}()

	// ── Gin router ────────────────────────────────────────────────────────────
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery())

	r.GET("/health", func(c *gin.Context) {
		c.JSON(200, gin.H{"status": "ok", "service": "api", "time": time.Now().UTC()})
	})

	// POST /infer — primary ingestion endpoint called by ML model wrappers
	r.POST("/infer", func(c *gin.Context) {
		start := time.Now()
		var payload InferencePayload
		if err := c.ShouldBindJSON(&payload); err != nil {
			apiRequests.WithLabelValues("/infer", "400").Inc()
			c.JSON(400, gin.H{"error": err.Error()})
			return
		}

		eventID := fmt.Sprintf("%s-%d", payload.ModelID, time.Now().UnixNano())
		event := map[string]interface{}{
			"event_id":       eventID,
			"model_id":       payload.ModelID,
			"model_version":  payload.ModelVersion,
			"timestamp":      time.Now().UTC(),
			"latency_ms":     payload.Latency,
			"input_features": payload.Features,
			"prediction":     payload.Prediction,
			"confidence":     payload.Confidence,
			"actual_label":   payload.ActualLabel,
			"error_type":     payload.ErrorType,
			"tags":           payload.Tags,
		}
		eventBytes, _ := json.Marshal(event)

		// Push to Kafka for ingester
		if err := producer.Produce(&kafka.Message{
			TopicPartition: kafka.TopicPartition{
				Topic:     &kafkaTopic,
				Partition: kafka.PartitionAny,
			},
			Value: eventBytes,
			Key:   []byte(payload.ModelID),
		}, nil); err != nil {
			apiRequests.WithLabelValues("/infer", "500").Inc()
			c.JSON(500, gin.H{"error": "kafka produce failed"})
			return
		}

		// If prompt/response present, queue for LLM evaluation
		if payload.Prompt != "" && payload.Response != "" {
			taskType := payload.TaskType
			if taskType == "" {
				taskType = "qa"
			}
			evalSample := map[string]string{
				"event_id":     eventID,
				"model_id":     payload.ModelID,
				"prompt":       payload.Prompt,
				"response":     payload.Response,
				"ground_truth": payload.GroundTruth,
				"task_type":    taskType,
				"timestamp":    time.Now().UTC().Format(time.RFC3339),
			}
			sampleBytes, _ := json.Marshal(evalSample)
			rdb.RPush(ctx, "eval:queue", string(sampleBytes))
		}

		apiRequests.WithLabelValues("/infer", "202").Inc()
		apiLatency.WithLabelValues("/infer").Observe(float64(time.Since(start).Milliseconds()))
		c.JSON(202, gin.H{"event_id": eventID, "status": "accepted"})
	})

	// GET /models — list active model IDs seen in Redis
	r.GET("/models", func(c *gin.Context) {
		keys, _ := rdb.Keys(ctx, "model:*").Result()
		c.JSON(200, gin.H{"models": keys})
	})

	// GET /summary — quick dashboard data
	r.GET("/summary", func(c *gin.Context) {
		queueDepth, _ := rdb.LLen(ctx, "eval:queue").Result()
		c.JSON(200, gin.H{
			"eval_queue_depth": queueDepth,
			"time":             time.Now().UTC(),
			"docs":             "See /metrics for Prometheus scrape endpoint",
		})
	})

	srv := &http.Server{Addr: ":" + apiPort, Handler: r}
	go func() {
		log.Printf("API server on :%s", apiPort)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("API server: %v", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	log.Println("API server shutting down...")
	shutCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	srv.Shutdown(shutCtx)
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
