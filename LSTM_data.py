import os
import pandas as pd


def filter_duplicate_rows(data):
    time_threshold = 1000
    filtered_data = [data.iloc[0]]

    for idx in range(1, len(data)):
        row = data.iloc[idx]
        prev_row = data.iloc[idx - 1]

        if row['time'] > time_threshold and \
                row['experience_diff'] == prev_row['experience_diff'] and \
                row['kda_diff'] == prev_row['kda_diff']:
            continue
        else:
            filtered_data.append(row)

    return pd.DataFrame(filtered_data)


# 设置输入和输出文件夹路径
input_folder = 'H:/final_data/chulidata2'
output_folder = 'H:/final_data/processed_data'

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的每个CSV文件
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 加载CSV文件
        df = pd.read_csv(os.path.join(input_folder, filename))

        # 将每个时间点的天辉队和夜魇队数据分开
        radiant_data = df[df['team'] == 'Radiant']
        dire_data = df[df['team'] == 'Dire']

        # 计算每个队伍的经济差、经验差和KDA差
        economy_diff = radiant_data.groupby('time')['gold'].sum() - dire_data.groupby('time')['gold'].sum()
        experience_diff = radiant_data.groupby('time')['current_XP'].sum() - dire_data.groupby('time')[
            'current_XP'].sum()
        economy_diff.index.name = 'time'
        experience_diff.index.name = 'time'

        radiant_kda = (radiant_data.groupby('time')['kill'].sum() + radiant_data.groupby('time')['assist'].sum()) / \
                      radiant_data.groupby('time')['death'].sum().replace(0, 1)
        dire_kda = (dire_data.groupby('time')['kill'].sum() + dire_data.groupby('time')['assist'].sum()) / \
                   dire_data.groupby('time')['death'].sum().replace(0, 1)
        kda_diff = radiant_kda - dire_kda
        kda_diff.index.name = 'time'

        # 添加胜利列
        winning_team = 1 if df['winner'].iloc[0] == 'Radiant' else -1

        data = pd.DataFrame({
            'time': economy_diff.index,
            'economy_diff': economy_diff.values,
            'experience_diff': experience_diff.values,
            'kda_diff': kda_diff.values,
            'winning_team': winning_team,
        })

        # 过滤重复的行
        filtered_data = filter_duplicate_rows(data)

        print(filtered_data)

        # 将数据保存到新的CSV文件中
        output_filename = os.path.join(output_folder, filename)
        filtered_data.to_csv(output_filename, index=False)
