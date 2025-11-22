#!/usr/bin/env python3
"""Simple script to consume PATE results from Kafka topic 'model_pate'"""
from kafka import KafkaConsumer
import json
import sys

def get_results():
    """Consume results from Kafka topic 'model_pate'"""
    print("Connecting to Kafka at localhost:29092...")
    print("Consuming results from topic 'model_pate'...")
    print("Press Ctrl+C to stop\n")

    consumer = KafkaConsumer(
        'model_pate',
        bootstrap_servers=['localhost:29092'],
        auto_offset_reset='earliest',  # Start from beginning
        value_deserializer=lambda x: json.loads(x.decode('utf-8')) if x else None,
        consumer_timeout_ms=10000,  # Wait up to 10 seconds for messages
    )

    results = []
    try:
        for message in consumer:
            if message.key:
                sample_id = message.key.decode('utf-8') if isinstance(message.key, bytes) else str(message.key)
            else:
                sample_id = "unknown"

            label = message.value
            results.append((sample_id, label))
            print(f"Sample {sample_id}: Label {label}")

        if not results:
            print("\nNo results found. The topic might be empty or already consumed.")
            print("Note: Kafka topics are consumed once per consumer group.")
            print("If you already consumed these results, they won't appear again.")
        else:
            print(f"\n\nTotal results received: {len(results)}")
            print("\nFirst 10 results:")
            for sample_id, label in results[:10]:
                print(f"  Sample {sample_id}: {label}")

            # Save to file
            output_file = "pate_results.csv"
            with open(output_file, 'w') as f:
                f.write("sample_id,label\n")
                for sample_id, label in results:
                    f.write(f"{sample_id},{label}\n")
            print(f"\nResults saved to: {output_file}")

    except KeyboardInterrupt:
        print("\n\nStopped by user")
        if results:
            print(f"Retrieved {len(results)} results before stopping")
    finally:
        consumer.close()

if __name__ == "__main__":
    get_results()
