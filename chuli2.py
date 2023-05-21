import os
import csv

def process_csv_file(file_path):
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # 跳过标题行
        team1_data = []
        team2_data = []

        for row in csv_reader:
            reduced_row = [row[0], row[3], row[14], row[15], row[18], row[19], row[20], row[21], row[22], row[23]]
            if row[3].strip() == 'Radiant':
                team1_data.append(reduced_row)
            else:
                team2_data.append(reduced_row)

    return team1_data, team2_data

def calculate_differences(team1_data, team2_data):
    differences = []
    for i in range(0, len(team1_data), 5):
        team1_sum = [0, 0, 0, 0, 0, 0, 0]
        team2_sum = [0, 0, 0, 0, 0, 0, 0]
        for j in range(5):
            team1_sum[0] += int(float(team1_data[i+j][2]))  # gold
            team1_sum[1] += int(float(team1_data[i+j][4]))  # kill
            team1_sum[2] += int(float(team1_data[i+j][5])) # death
            team1_sum[3] += int(float(team1_data[i+j][3]))# LH
            team1_sum[4] += int(float(team1_data[i+j][6]))  # assist
            team1_sum[5] += int(float(team1_data[i+j][7]))  # rest_tower
            team1_sum[6] += int(float(team1_data[i+j][8]))  # rest_barracks

            team2_sum[0] += int(float(team2_data[i+j][2]))  # gold
            team2_sum[1] += int(float(team2_data[i+j][4]))  # kill
            team2_sum[2] += int(float(team2_data[i+j][5]))  # death
            team2_sum[3] += int(float(team2_data[i+j][3]))  # LH
            team2_sum[4] += int(float(team2_data[i+j][6]))  # assist
            team2_sum[5] += int(float(team2_data[i+j][7]))  # rest_tower
            team2_sum[6] += int(float(team2_data[i+j][8]))  # rest_barracks

        differences.append([team1_sum[i] - team2_sum[i] for i in range(7)])

    return differences

def save_differences_to_csv(differences, output_file):
    with open(output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Time', 'Gold Diff', 'Kill Diff', 'Death Diff', 'LH Diff', 'Assist Diff', 'Tower Diff', 'Barracks Diff'])
        for i, diff in enumerate(differences):
            csv_writer.writerow([i*5, diff[0], diff[1], diff[2], diff[3], diff[4], diff[5], diff[6]])

def add_winner_and_time_to_csv(input_folder_path):
    for file in os.listdir(input_folder_path):
        if file.endswith('.csv'):
            csv
