#!/bin/bash
# Wait for Kafka and create topics
set -e
KAFKA_CONTAINER="${KAFKA_CONTAINER:-$(docker compose -f docker/docker-compose.yml ps -q kafka)}"
echo "Waiting for Kafka..."
until docker exec "$KAFKA_CONTAINER" kafka-broker-api-versions \
  --bootstrap-server localhost:9092 &>/dev/null; do
  sleep 2
done
echo "Kafka ready. Creating topics..."
docker exec "$KAFKA_CONTAINER" kafka-topics \
  --bootstrap-server localhost:9092 \
  --create --if-not-exists \
  --topic ml-inference-events \
  --partitions 6 \
  --replication-factor 1
echo "Topics created."
