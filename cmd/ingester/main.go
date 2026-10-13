// cmd/ingester/main.go
// Kafka consumer that ingests raw ML inference telemetry events,
// decorates them with context, detects drift, and exposes Prometheus metrics.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/confluentinc/confluent-kafka-go/v2/kafka"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// InferenceEvent represents a single ML model inference captured from production.
type InferenceEvent struct {
	EventID       string             `json:"event_id"`
	ModelID       string             `json:"model_id"`
	ModelVersion  string             `json:"model_version"`
	Timestamp     time.Time          `json:"timestamp"`
	Latency       float64            `json:"latency_ms"`
	InputFeatures map[string]float64 `json:"input_features"`
	Prediction    float64            `json:"prediction"`
	Confidence    float64            `json:"confidence"`
	ActualLabel   *float64           `json:"actual_label,omitempty"`
	ErrorType     string             `json:"error_type,omitempty"`
	Tags          map[string]string  `json:"tags"`
}

// DriftWindow holds a rolling window of feature values for drift computation.
type DriftWindow struct {
	values    []float64
	maxSize   int
	baseline  []float64 // reference distribution (from training data)
}

func newDriftWindow(baseline []float64, size int) *DriftWindow {
	return &DriftWindow{
		values:   make([]float64, 0, size),
		maxSize:  size,
		baseline: baseline,
	}
}

func (d *DriftWindow) add(v float64) {
	if len(d.values) >= d.maxSize {
		d.values = d.values[1:]
	}
	d.values = append(d.values, v)
}

// PSIScore computes Population Stability Index between baseline and current window.
// PSI < 0.1 = stable, 0.1-0.2 = slight drift, > 0.2 = significant drift.
func (d *DriftWindow) PSIScore() float64 {
	if len(d.values) < 10 || len(d.baseline) < 10 {
		return 0
	}
	buckets := 10
	min, max := d.baseline[0], d.baseline[0]
	for _, v := range d.baseline {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	if max == min {
		return 0
	}
	bucketWidth := (max - min) / float64(buckets)
	baselineCounts := make([]float64, buckets)
	currentCounts := make([]float64, buckets)
	for _, v := range d.baseline {
		idx := int((v - min) / bucketWidth)
		if idx >= buckets {
			idx = buckets - 1
		}
		baselineCounts[idx]++
	}
	for _, v := range d.values {
		idx := int((v - min) / bucketWidth)
		if idx < 0 {
			idx = 0
		}
		if idx >= buckets {
			idx = buckets - 1
		}
		currentCounts[idx]++
	}
	psi := 0.0
	for i := 0; i < buckets; i++ {
		bp := (baselineCounts[i] + 0.0001) / float64(len(d.baseline))
		cp := (currentCounts[i] + 0.0001) / float64(len(d.values))
		psi += (cp - bp) * math.Log(cp/bp)
	}
	return psi
}

func main() {
	kafkaBroker := envOrDefault("KAFKA_BROKER", "localhost:9092")
	kafkaTopic := envOrDefault("KAFKA_TOPIC", "ml-inference-events")
	metricsPort := envOrDefault("METRICS_PORT", "2112")

	// ── Prometheus metrics ──────────────────────────────────────────────────
	inferenceTotal := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "ml_inference_total",
		Help: "Total inference events processed, by model and version.",
	}, []string{"model_id", "model_version"})

	inferenceLatency := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "ml_inference_latency_ms",
		Help:    "Inference latency distribution in milliseconds.",
		Buckets: []float64{10, 25, 50, 100, 200, 500, 1000, 2000, 5000},
	}, []string{"model_id", "model_version"})

	inferenceErrors := prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "ml_inference_errors_total",
		Help: "Total inference errors by model and error type.",
	}, []string{"model_id", "error_type"})

	modelConfidence := prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "ml_model_confidence",
		Help:    "Distribution of model prediction confidence scores.",
		Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
	}, []string{"model_id"})

	featureDriftPSI := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "ml_feature_drift_psi",
		Help: "Population Stability Index per feature. PSI>0.2 = significant drift.",
	}, []string{"model_id", "feature_name"})

	predictionDrift := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "ml_prediction_drift_psi",
		Help: "PSI of prediction distribution vs baseline.",
	}, []string{"model_id"})

	modelAccuracy := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "ml_model_accuracy",
		Help: "Rolling accuracy where ground truth labels are available.",
	}, []string{"model_id"})

	kafkaLag := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "kafka_consumer_lag",
		Help: "Kafka consumer lag (messages behind) per partition.",
	}, []string{"topic", "partition"})

	reg := prometheus.NewRegistry()
	reg.MustRegister(inferenceTotal, inferenceLatency, inferenceErrors,
		modelConfidence, featureDriftPSI, predictionDrift, modelAccuracy, kafkaLag)

	// ── Drift state (per model, per feature) ────────────────────────────────
	// Baselines here are synthetic; in production, load from your feature store.
	baselineFeatures := map[string][]float64{
		"feature_0": generateBaseline(500, 0.5, 0.1),
		"feature_1": generateBaseline(500, 2.3, 0.4),
		"feature_2": generateBaseline(500, 0.0, 1.0),
	}
	driftWindows := map[string]map[string]*DriftWindow{}
	predDriftWindows := map[string]*DriftWindow{}
	accuracyTrackers := map[string]*accuracyTracker{}

	// ── Kafka consumer ───────────────────────────────────────────────────────
	c, err := kafka.NewConsumer(&kafka.ConfigMap{
		"bootstrap.servers":  kafkaBroker,
		"group.id":           "ml-observability-ingester",
		"auto.offset.reset":  "earliest",
		"enable.auto.commit": true,
	})
	if err != nil {
		log.Fatalf("Failed to create Kafka consumer: %v", err)
	}
	defer c.Close()

	if err := c.SubscribeTopics([]string{kafkaTopic}, nil); err != nil {
		log.Fatalf("Failed to subscribe: %v", err)
	}
	log.Printf("Ingester started. Consuming from %s/%s", kafkaBroker, kafkaTopic)

	// ── Metrics HTTP server ──────────────────────────────────────────────────
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, `{"status":"ok","service":"ingester"}`)
	})
	go func() {
		log.Printf("Metrics server on :%s", metricsPort)
		if err := http.ListenAndServe(":"+metricsPort, mux); err != nil {
			log.Fatalf("Metrics server failed: %v", err)
		}
	}()

	// ── Drift ticker (recompute PSI every 30s) ───────────────────────────────
	driftTicker := time.NewTicker(30 * time.Second)
	go func() {
		for range driftTicker.C {
			for modelID, featureMap := range driftWindows {
				for featureName, window := range featureMap {
					psi := window.PSIScore()
					featureDriftPSI.WithLabelValues(modelID, featureName).Set(psi)
					if psi > 0.2 {
						log.Printf("DRIFT ALERT model=%s feature=%s PSI=%.4f", modelID, featureName, psi)
					}
				}
				if pw, ok := predDriftWindows[modelID]; ok {
					predictionDrift.WithLabelValues(modelID).Set(pw.PSIScore())
				}
				if at, ok := accuracyTrackers[modelID]; ok {
					modelAccuracy.WithLabelValues(modelID).Set(at.accuracy())
				}
			}
		}
	}()

	// ── Graceful shutdown ────────────────────────────────────────────────────
	ctx, cancel := context.WithCancel(context.Background())
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		log.Println("Shutting down ingester...")
		cancel()
	}()

	// ── Main consume loop ────────────────────────────────────────────────────
	for {
		select {
		case <-ctx.Done():
			driftTicker.Stop()
			return
		default:
		}

		msg, err := c.ReadMessage(500 * time.Millisecond)
		if err != nil {
			if kafkaErr, ok := err.(kafka.Error); ok && kafkaErr.Code() == kafka.ErrTimedOut {
				continue
			}
			log.Printf("Kafka read error: %v", err)
			continue
		}

		var event InferenceEvent
		if err := json.Unmarshal(msg.Value, &event); err != nil {
			log.Printf("Unmarshal error: %v", err)
			inferenceErrors.WithLabelValues("unknown", "parse_error").Inc()
			continue
		}

		// Record core metrics
		inferenceTotal.WithLabelValues(event.ModelID, event.ModelVersion).Inc()
		inferenceLatency.WithLabelValues(event.ModelID, event.ModelVersion).Observe(event.Latency)
		modelConfidence.WithLabelValues(event.ModelID).Observe(event.Confidence)
		if event.ErrorType != "" {
			inferenceErrors.WithLabelValues(event.ModelID, event.ErrorType).Inc()
		}

		// Update drift windows
		if _, ok := driftWindows[event.ModelID]; !ok {
			driftWindows[event.ModelID] = map[string]*DriftWindow{}
			predDriftWindows[event.ModelID] = newDriftWindow(
				generateBaseline(500, 0.5, 0.15), 200)
			accuracyTrackers[event.ModelID] = &accuracyTracker{}
		}
		for feat, val := range event.InputFeatures {
			if _, ok := driftWindows[event.ModelID][feat]; !ok {
				baseline := baselineFeatures[feat]
				if baseline == nil {
					baseline = generateBaseline(500, 0.5, 0.2)
				}
				driftWindows[event.ModelID][feat] = newDriftWindow(baseline, 200)
			}
			driftWindows[event.ModelID][feat].add(val)
		}
		predDriftWindows[event.ModelID].add(event.Prediction)
		if event.ActualLabel != nil {
			accuracyTrackers[event.ModelID].record(event.Prediction, *event.ActualLabel)
		}

		// Kafka lag (best-effort)
		tp := kafka.TopicPartition{Topic: &kafkaTopic, Partition: msg.TopicPartition.Partition}
		if positions, err := c.Position([]kafka.TopicPartition{tp}); err == nil {
			if md, err := c.GetMetadata(&kafkaTopic, false, 2000); err == nil {
				if topicMD, ok := md.Topics[kafkaTopic]; ok {
					for _, partMD := range topicMD.Partitions {
						if partMD.ID == msg.TopicPartition.Partition {
							for _, pos := range positions {
								lag := int64(partMD.ID) - int64(pos.Offset)
								if lag < 0 {
									lag = 0
								}
								kafkaLag.WithLabelValues(kafkaTopic,
									fmt.Sprintf("%d", partMD.ID)).Set(float64(lag))
							}
						}
					}
				}
			}
		}

		log.Printf("Processed event_id=%s model=%s latency=%.1fms confidence=%.3f",
			event.EventID, event.ModelID, event.Latency, event.Confidence)
	}
}

// ── Helpers ──────────────────────────────────────────────────────────────────

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// generateBaseline produces a synthetic Gaussian baseline for demo / dev use.
// In production replace with actual training-data statistics.
func generateBaseline(n int, mean, stddev float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		// Box-Muller
		u := float64(i+1) / float64(n+1)
		v := float64((i*7+3)%n+1) / float64(n+1)
		z := math.Sqrt(-2*math.Log(u)) * math.Cos(2*math.Pi*v)
		out[i] = mean + stddev*z
	}
	return out
}

type accuracyTracker struct {
	correct int
	total   int
}

func (a *accuracyTracker) record(pred, actual float64) {
	a.total++
	// Binary classification: round prediction to 0 or 1
	if math.Round(pred) == math.Round(actual) {
		a.correct++
	}
}

func (a *accuracyTracker) accuracy() float64 {
	if a.total == 0 {
		return 0
	}
	return float64(a.correct) / float64(a.total)
}
