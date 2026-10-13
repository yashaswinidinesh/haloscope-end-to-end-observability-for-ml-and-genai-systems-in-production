#!/usr/bin/env python3
"""
load_simulator.py
Simulates ML model inference traffic by POSTing events to the API gateway.
Includes normal traffic, degraded latency spikes, confidence drops,
feature drift injection, and LLM response samples for the evaluator.

Usage:
    python scripts/load_simulator.py [--api http://localhost:8080] [--rate 10]
"""
import argparse
import json
import math
import random
import sys
import time
import uuid
from datetime import datetime, timezone

import requests

MODELS = [
    {"id": "recommendation-v2", "version": "2.1.0", "type": "classification"},
    {"id": "fraud-detector-v1", "version": "1.4.2", "type": "classification"},
    {"id": "llm-summarizer",    "version": "0.9.1", "type": "qa"},
]

LLM_PROMPTS = [
    ("Summarize the key risks of investing in volatile markets.",
     "Volatile markets carry risks including rapid price swings, margin calls, and emotional decision-making leading to poor timing. Diversification and stop-loss orders help mitigate exposure.",
     "Key risks include price volatility, liquidity issues, and behavioral biases."),
    ("What is the capital of Australia?",
     "The capital of Australia is Canberra.",
     "Canberra"),
    ("Explain gradient descent in one sentence.",
     "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction that minimizes the loss function.",
     "An iterative optimization algorithm minimizing loss by following the negative gradient."),
    ("What year did World War II end?",
     "World War II ended in 1945.",  # correct
     "1945"),
    ("What year did World War II end?",
     "World War II ended in 1942.",  # hallucination injected
     "1945"),
]

def make_features(drift=False):
    base = {
        "feature_0": random.gauss(0.5, 0.1),
        "feature_1": random.gauss(2.3, 0.4),
        "feature_2": random.gauss(0.0, 1.0),
    }
    if drift:
        # Shift distribution — simulates covariate shift
        base["feature_0"] += random.gauss(0.8, 0.05)  # mean shift
        base["feature_1"] *= random.uniform(1.5, 2.0)  # scale drift
    return base

def make_event(model, drift=False, latency_spike=False, low_confidence=False):
    latency = random.gauss(120, 20)
    if latency_spike:
        latency = random.gauss(900, 100)

    confidence = random.uniform(0.75, 0.99)
    if low_confidence:
        confidence = random.uniform(0.45, 0.6)

    prediction = round(random.random())
    actual = prediction if random.random() > 0.1 else 1 - prediction  # 10% error rate

    event = {
        "model_id":      model["id"],
        "model_version": model["version"],
        "latency_ms":    round(max(latency, 5), 2),
        "input_features": make_features(drift=drift),
        "prediction":    prediction,
        "confidence":    round(confidence, 4),
        "actual_label":  float(actual),
        "tags": {
            "environment": "production",
            "region":      "us-west-2",
            "experiment":  "control" if not drift else "drift_test",
        },
    }

    # For LLM models, attach prompt/response for LLM-as-judge eval
    if model["type"] == "qa":
        prompt, response, ground_truth = random.choice(LLM_PROMPTS)
        event["prompt"] = prompt
        event["response"] = response
        event["ground_truth"] = ground_truth
        event["task_type"] = "qa"
        event.pop("input_features", None)
        event.pop("actual_label", None)
        event["prediction"] = 0.0
        event["confidence"] = random.uniform(0.7, 0.95)

    return event

def run(api_base: str, rate: float, drift_after: int, spike_after: int):
    print(f"Simulator → {api_base}  rate={rate}/s  drift_after={drift_after}s  spike_after={spike_after}s")
    url = f"{api_base}/infer"
    sent = 0
    start = time.time()

    while True:
        elapsed = time.time() - start
        drift = elapsed > drift_after
        spike = (elapsed % (spike_after * 2)) < 30  # spike for 30s every spike_after*2 seconds

        model = random.choice(MODELS)
        event = make_event(
            model,
            drift=drift and model["id"] == "recommendation-v2",
            latency_spike=spike and model["id"] == "fraud-detector-v1",
            low_confidence=drift and model["id"] == "fraud-detector-v1",
        )

        try:
            resp = requests.post(url, json=event, timeout=5)
            sent += 1
            if resp.status_code == 202:
                if sent % 50 == 0:
                    print(f"[{elapsed:.0f}s] Sent {sent} events  drift={drift}  spike={spike}")
            else:
                print(f"Unexpected status {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        except requests.RequestException as e:
            print(f"Request error: {e}", file=sys.stderr)

        time.sleep(1.0 / max(rate, 0.1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML inference load simulator")
    parser.add_argument("--api", default="http://localhost:8080", help="API gateway base URL")
    parser.add_argument("--rate", type=float, default=5.0, help="Events per second")
    parser.add_argument("--drift-after", type=int, default=120, help="Seconds before injecting drift")
    parser.add_argument("--spike-after", type=int, default=90, help="Seconds before latency spikes")
    args = parser.parse_args()
    run(args.api, args.rate, args.drift_after, args.spike_after)
