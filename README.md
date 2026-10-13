# Haloscope

End-to-end observability for ML and GenAI systems in production.

Captures model inputs, predictions, latency, data drift, and LLM response quality. Gives teams the dashboards, alerts, and evaluation primitives to ship AI-powered products with confidence — without instrumenting every model individually.

## Contents

- [Architecture](#architecture)
- [Services and stack](#tools-used)
- [Metrics](#metrics-exposed)
- [Quickstart](#quickstart)
- [LLM evaluation](#5-configure-llm-as-judge)
- [Local development](#local-development-without-docker)
- [Alerts](#alert-rules)
- [Instrumenting a model](#instrumenting-your-own-model)
- [Project structure](#project-structure)
- [What this demonstrates](#what-this-demonstrates)

---

## Architecture

```
ML Model / Service
        │  POST /infer
        ▼
  ┌─────────────┐        ┌──────────────────┐
  │  API Gateway│──────▶│  Kafka            │
  │  (Go + Gin) │        │  ml-inference-    │
  └──────┬──────┘        │  events           │
         │               └────────┬─────────┘
         │ eval samples           │ consume
         ▼                        ▼
  ┌─────────────┐        ┌──────────────────────────────────┐
  │   Redis     │        │  Ingester (Go)                   │
  │  eval:queue │        │  • PSI drift detection per feature│
  └──────┬──────┘        │  • Accuracy tracking             │
         │               │  • Kafka lag monitoring          │
         ▼               │  → Prometheus metrics :2112      │
  ┌─────────────────┐    └──────────────────────────────────┘
  │ Evaluator (Go)  │
  │ • LLM-as-judge  │    ┌──────────────────────────────────┐
  │ • Grounding     │    │  Prometheus → Grafana Dashboard  │
  │ • Hallucination │    │  • Latency SLOs (p50/p95/p99)   │
  │ • Task success  │───▶│  • Feature drift PSI             │
  │ → metrics :2113 │    │  • LLM eval scores               │
  └─────────────────┘    │  • Accuracy & confidence         │
                         └──────────────────────────────────┘

  Distributed tracing: OpenTelemetry → Tempo → Grafana
  Logs: Promtail → Loki → Grafana
```

## Tools Used

| Tool | Purpose |
|------|---------|
| **Kafka** | Telemetry ingestion — decouples model instrumentation from processing |
| **Prometheus** | Metrics store — drift PSI, latency percentiles, accuracy, hallucination rate |
| **Grafana** | Dashboards + alerts — auto-provisioned ML overview dashboard |
| **Loki** | Log aggregation — correlate anomalies with service logs |
| **Tempo** | Distributed tracing — end-to-end request traces via OpenTelemetry |
| **OpenTelemetry** | Trace and metric collection — vendor-neutral instrumentation |
| **Redis** | Eval queue + rolling state for hallucination windows |
| **Go** | API gateway, Kafka ingester, and asynchronous evaluator |
| **Python** | Configurable inference traffic simulator |
| **Docker Compose** | Local orchestration for the complete stack |

## Services

| Service | Language | Port | Responsibility |
|---------|----------|------|---------------|
| `api` | Go | 8080 | REST gateway — receives `/infer` events, produces to Kafka, queues eval samples |
| `ingester` | Go | 2112 | Kafka consumer — PSI drift detection, accuracy tracking, Prometheus metrics |
| `evaluator` | Go | 2113 | LLM-as-judge — pops eval queue from Redis, scores response quality |

## Metrics Exposed

### Model Performance
| Metric | Type | Description |
|--------|------|-------------|
| `ml_inference_total` | Counter | Total inferences by model and version |
| `ml_inference_latency_ms` | Histogram | Latency distribution — use for p50/p95/p99 SLOs |
| `ml_inference_errors_total` | Counter | Errors by model and error type |
| `ml_model_confidence` | Histogram | Prediction confidence distribution |
| `ml_model_accuracy` | Gauge | Rolling accuracy where ground truth is available |

### Drift Detection (PSI)
| Metric | Type | Description |
|--------|------|-------------|
| `ml_feature_drift_psi` | Gauge | Per-feature PSI vs training baseline. >0.1 = slight, >0.2 = significant |
| `ml_prediction_drift_psi` | Gauge | Prediction distribution drift vs baseline |

### LLM / GenAI Evaluation
| Metric | Type | Description |
|--------|------|-------------|
| `llm_eval_score` | Histogram | Overall quality score from LLM judge (0–1) |
| `llm_eval_grounding` | Histogram | Factual accuracy / citation quality |
| `llm_eval_coherence` | Histogram | Logical flow of response |
| `llm_eval_hallucination_rate` | Gauge | Rolling hallucination rate (0=clean, 1=hallucinated) |
| `llm_eval_total` | Counter | Evaluations run — track throughput and error rate |
| `eval_queue_depth` | Gauge | Redis queue depth — detect evaluator backpressure |

### Infrastructure
| Metric | Type | Description |
|--------|------|-------------|
| `kafka_consumer_lag` | Gauge | Messages behind per partition |
| `api_requests_total` | Counter | API request count by endpoint and status |
| `api_request_latency_ms` | Histogram | API gateway latency |

---

## Quickstart

### Prerequisites

- Docker Desktop (or Docker Engine) with Docker Compose v2
- Go 1.22+ (for local development)
- Python 3.10+ (for the load simulator)
- **For LLM evaluation** (pick one):
  - [Ollama](https://ollama.ai) running locally (`ollama pull llama3.1:8b`)
  - OpenAI API key

### 1. Start the full stack

```bash
git clone https://github.com/yashaswinidinesh/haloscope.git
cd haloscope

# Build the Go services and start the full stack
docker compose -f docker/docker-compose.yml up -d --build

# Wait for Kafka and create the inference topic
bash scripts/create_topics.sh
```

### 2. Verify everything is up

```bash
# Check all containers
docker compose -f docker/docker-compose.yml ps

# API health
curl http://localhost:8080/health

# Prometheus targets (all should be UP)
open http://localhost:9090/targets

# Grafana (local default login: admin / admin)
open 'http://localhost:3000/d/ml-obs-overview/haloscope-e28094-ml-observability-overview?orgId=1&refresh=10s'

# Kafka UI
open http://localhost:8090
```

Change Grafana's default password before exposing the service outside a local development environment.

### 3. Send your first inference event

```bash
# Classification model event
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud-detector-v1",
    "model_version": "1.4.2",
    "latency_ms": 87.3,
    "input_features": {"feature_0": 0.52, "feature_1": 2.1, "feature_2": -0.3},
    "prediction": 0,
    "confidence": 0.91,
    "actual_label": 0,
    "tags": {"environment": "production", "region": "us-west-2"}
  }'

# LLM model event (queued for LLM-as-judge evaluation)
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "llm-summarizer",
    "model_version": "0.9.1",
    "latency_ms": 420,
    "prediction": 0,
    "confidence": 0.88,
    "prompt": "Summarize the key benefits of transformer architectures.",
    "response": "Transformers enable parallel processing of sequences and capture long-range dependencies via self-attention, overcoming RNN limitations.",
    "ground_truth": "Key benefits: parallelism, long-range dependency capture, and scalability.",
    "task_type": "qa",
    "tags": {"environment": "production"}
  }'
```

### 4. Run the load simulator

Generates continuous traffic with drift injection and latency spikes after configurable delays:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install requests
.venv/bin/python scripts/load_simulator.py --api http://localhost:8080 --rate 5 --drift-after 120 --spike-after 90
```

The simulator continuously sends classification and LLM events. After the configured delay it injects feature drift and periodic latency spikes. The drift alert fires only after its configured sustained-duration window; inspect the live PSI metric in Grafana while the alert condition matures.

### 5. Configure LLM-as-judge

**Ollama (free, local):**
```bash
ollama pull llama3.1:8b
# The evaluator defaults to http://host.docker.internal:11434 — no config needed
```

**OpenAI:**
```bash
# Put these values in a project-root .env file (already ignored by Git):
JUDGE_ENDPOINT=https://api.openai.com/v1/chat/completions
JUDGE_MODEL=gpt-4o-mini
JUDGE_API_KEY=your-api-key
```

Never commit real API keys. Compose reads the project-root `.env` file when interpolating the evaluator's environment variables.

---

## Local Development (without Docker)

```bash
# Start dependencies only
docker compose -f docker/docker-compose.yml up -d zookeeper kafka redis prometheus grafana loki tempo otel-collector
bash scripts/create_topics.sh

# Run services locally
KAFKA_BROKER=localhost:9092 REDIS_ADDR=localhost:6379 go run ./cmd/api &
KAFKA_BROKER=localhost:9092 go run ./cmd/ingester &
REDIS_ADDR=localhost:6379 go run ./cmd/evaluator &

# Simulator
.venv/bin/python scripts/load_simulator.py
```

Run `go test ./...` from the repository root to compile and test all Go packages.

---

## Alert Rules

Pre-configured Prometheus alert rules fire on:

| Alert | Condition |
|-------|-----------|
| `ModelLatencyHigh` | P95 latency > 500ms for 2m |
| `ModelLatencyCritical` | P99 latency > 2000ms for 1m |
| `FeatureDriftDetected` | PSI > 0.2 for any feature, 5m |
| `PredictionDriftDetected` | Prediction PSI > 0.25 for 5m |
| `ModelAccuracyDegraded` | Accuracy < 80% for 10m |
| `ModelAccuracyCritical` | Accuracy < 65% for 5m |
| `HallucinationRateHigh` | Hallucination rate > 15% for 5m |
| `LLMQualityDegraded` | Median eval score < 0.6 for 10m |
| `InferenceErrorRateHigh` | Error rate > 5% for 3m |
| `KafkaConsumerLagHigh` | Lag > 1000 messages for 5m |

---

## Instrumenting Your Own Model

Wrap any model with a single POST call:

```python
import requests, time

def observed_predict(model, input_features, ground_truth=None):
    start = time.time()
    result = model.predict(input_features)
    latency_ms = (time.time() - start) * 1000

    requests.post("http://localhost:8080/infer", json={
        "model_id":      model.id,
        "model_version": model.version,
        "latency_ms":    latency_ms,
        "input_features": {f"feature_{i}": float(v) for i, v in enumerate(input_features)},
        "prediction":    float(result.label),
        "confidence":    float(result.confidence),
        "actual_label":  float(ground_truth) if ground_truth is not None else None,
        "tags":          {"environment": "production"},
    }, timeout=2)
    return result
```

For LLM models, add `prompt`, `response`, `ground_truth`, and `task_type` fields.

---

## Project Structure

```
./
├── cmd/
│   ├── api/main.go          # REST gateway — Gin, Kafka producer, Redis
│   ├── ingester/main.go     # Kafka consumer — PSI drift, accuracy, Prometheus
│   └── evaluator/main.go    # LLM-as-judge — Redis queue, OpenAI/Ollama
├── docker/
│   ├── docker-compose.yml   # Full stack: Kafka, Redis, Prometheus, Grafana, Loki, Tempo, OTel
│   ├── prometheus.yml       # Scrape config for all services
│   ├── alerts.yml           # ML-specific alert rules
│   ├── tempo.yaml           # Distributed tracing config
│   ├── otel-collector.yaml  # OpenTelemetry collector pipeline
│   ├── promtail.yml         # Log shipping to Loki
│   ├── Dockerfile.api
│   ├── Dockerfile.ingester
│   ├── Dockerfile.evaluator
│   └── grafana/provisioning/
│       ├── datasources/     # Prometheus, Loki, Tempo auto-provisioned
│       └── dashboards/      # ML overview dashboard auto-provisioned
├── scripts/
│   ├── load_simulator.py    # Traffic generator with drift and spike injection
│   └── create_topics.sh     # Kafka topic bootstrap
└── go.mod
```

---

## What This Demonstrates

- **Observability-by-default** — every model that calls `/infer` gets latency, error, drift, and accuracy metrics automatically, without per-model instrumentation
- **PSI-based drift detection** — Population Stability Index computed in a rolling window against training baselines, per feature and per prediction distribution
- **LLM-as-judge evaluation** — async evaluation pipeline scoring grounding, coherence, relevance, and hallucination using a separate judge LLM
- **Full telemetry stack** — Kafka → Prometheus → Grafana (metrics), Loki (logs), Tempo + OpenTelemetry (distributed traces) wired together in Grafana's unified view
- **Reusable SDK pattern** — models are instrumented with a single HTTP call; the platform handles all processing, storage, and alerting
- **Production alert rules** — pre-calibrated SLO thresholds for latency, drift, accuracy, and hallucination rate

---

## Author

**Yashaswini Dinesh** · [LinkedIn](https://linkedin.com/in/yashaswini-dinesh-roopa) · [GitHub](https://github.com/yashaswinidinesh) · [Portfolio](https://yashaswinidineshportfolio.netlify.app)
