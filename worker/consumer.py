import os
import json
from kafka import KafkaConsumer
from worker.tasks import process_drift_batch

def start_consumer():
    broker = os.getenv("KAFKA_BROKER", "localhost:9092")
    topic = "dews-live-stream"
    
    print(f"Starting Kafka Consumer on {broker} for topic {topic}...", flush=True)
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=[broker],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        group_id="dews-group",
        auto_offset_reset="earliest",
    )
    
    batch = []
    batch_index = 0
    BATCH_SIZE = 80
    
    for message in consumer:
        batch.append(message.value)
        if len(batch) >= BATCH_SIZE:
            print(f"Submitting batch {batch_index} ({len(batch)} records) to Celery...", flush=True)
            process_drift_batch.delay(batch, batch_index)
            batch = []
            batch_index += 1

if __name__ == "__main__":
    start_consumer()
