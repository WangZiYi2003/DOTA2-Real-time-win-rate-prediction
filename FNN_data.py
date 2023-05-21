import os
import pandas as pd

def process_data(input_path, output_path):
    files = os.listdir(input_path)

    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(input_path, file)
            df = pd.read_csv(file_path)

            for i in range(1200, int(df['time'].max()), 30):
                rows_to_compare = df[df['time'] == i]

                if not rows_to_compare.empty:
                    previous_time = i - 30
                    previous_rows = df[df['time'] == previous_time]

                    if not previous_rows.empty:
                        first_row_current = rows_to_compare.iloc[0]
                        first_row_previous = previous_rows.iloc[0]

                        if (first_row_current['gold'] == first_row_previous['gold']) and (first_row_current['current_XP'] == first_row_previous['current_XP']):
                            df = df[df['time'] < i]
                            break

            output_file = os.path.join(output_path, file)
            df.to_csv(output_file, index=False)

input_path = "H:/final_data/chulidata2"
output_path = "H:/final_data/base_data"

if not os.path.exists(output_path):
    os.makedirs(output_path)

process_data(input_path, output_path)
