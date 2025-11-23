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

            # Calculate label distribution
            from collections import Counter
            label_counts = Counter([label for _, label in results])
            total_samples = len(results)

            print("\n" + "="*60)
            print("LABEL DISTRIBUTION STATISTICS")
            print("="*60)
            print(f"{'Label':<10} {'Count':<12} {'Percentage':<15} {'Bar'}")
            print("-"*60)

            # Sort by label number for better readability
            sorted_labels = sorted(label_counts.keys(), key=lambda x: int(x) if str(x).isdigit() else x)
            for label in sorted_labels:
                count = label_counts[label]
                percentage = (count / total_samples) * 100
                bar_length = int(percentage / 2)  # Scale bar to fit in terminal
                bar = "█" * bar_length
                print(f"{label:<10} {count:<12} {percentage:>6.2f}%{'':<6} {bar}")

            print("-"*60)
            print(f"{'Total':<10} {total_samples:<12} {'100.00%':<15}")
            print("="*60)

            # Check for balance (useful for MNIST which should be ~10% per label)
            expected_percentage = 100.0 / len(label_counts) if label_counts else 0
            print(f"\nExpected percentage per label (balanced): ~{expected_percentage:.2f}%")

            # Find most and least common labels
            most_common = label_counts.most_common(1)[0]
            least_common = label_counts.most_common()[-1]
            print(f"Most common label: {most_common[0]} ({most_common[1]} samples, {(most_common[1]/total_samples)*100:.2f}%)")
            print(f"Least common label: {least_common[0]} ({least_common[1]} samples, {(least_common[1]/total_samples)*100:.2f}%)")

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
