import pandas as pd
import os

# Define the directory where the CSV files are
directory = 'H:/final_data/chulidata2'

# Define time intervals
time_intervals = [(0,1200), (1200, 1500), (1500, 1800), (1800, 2100), (2100, 2400), (2400, 2700), (2700, 3000), (3000, float('inf'))]

# Initialize counts
time_counts = {interval: 0 for interval in time_intervals}

# Loop over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Load the CSV file
        df = pd.read_csv(os.path.join(directory, filename))

        # Extract the last non-null time
        last_time = df['time'].dropna().iloc[-1]

        # Update the time interval count
        for interval in time_intervals:
            if interval[0] <= last_time < interval[1]:
                time_counts[interval] += 1
                break

# Print the results
print("Time Interval Counts:")
for interval, count in time_counts.items():
    print(f"{interval[0]}-{interval[1]}: {count}")

