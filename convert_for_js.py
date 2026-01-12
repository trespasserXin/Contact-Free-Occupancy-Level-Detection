import csv
import json
from read_crossing_head import crossing_list

# ==========================================
# 1. PASTE YOUR ULTRASONIC LIST HERE
# ==========================================
# Example: List of seconds when the sensor triggered
ultrasonic_events = crossing_list

# ==========================================
# 2. CONFIGURATION
# ==========================================
CSV_FILENAME = 'compare_y_pred_y_true.csv'
TIME_INTERVAL = 5  # Your CSV rows are 5 seconds apart


def map_tier(val):
    """Maps 0->Low, 1->Med, 2->High"""
    mapping = {0: 'Low', 1: 'Med', 2: 'High'}
    return mapping.get(int(val), 'Unknown')


def process_data():
    output_data = []

    try:
        with open(CSV_FILENAME, 'r') as f:
            reader = csv.DictReader(f)

            # Ensure your CSV headers match these keys or update them here
            # Expected headers: 'time', 'true_head_count', 'pred_label', 'true_label'

            for row in reader:
                start_time = float(row['time'])
                end_time = start_time + TIME_INTERVAL

                # Check if an ultrasonic event happened in this 5-second window
                # Logic: Is there any event 'u' such that start <= u < end
                sensor_count = sum(1 for u in ultrasonic_events if start_time <= u < end_time)

                output_data.append({
                    "time": start_time,
                    "count": int(row['Head Count']),
                    "pred_tier": map_tier(row['y_pred']),
                    "true_tier": map_tier(row['y_true']),
                    "ultrasonic": sensor_count
                })

        # Print the result as JSON
        print("COPY EVERYTHING BELOW THIS LINE:")
        print("--------------------------------")
        with open('vis_program.json', 'w') as f:
            f.write(json.dumps(output_data, indent=2))
        # print(json.dumps(output_data, indent=2))

    except FileNotFoundError:
        print(f"Error: Could not find {CSV_FILENAME}")
    except KeyError as e:
        print(f"Error: Your CSV is missing the column: {e}")


if __name__ == "__main__":
    process_data()