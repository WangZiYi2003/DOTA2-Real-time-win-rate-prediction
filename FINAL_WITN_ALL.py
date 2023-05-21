import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os


# 数据预处理
def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    data = []
    for i in range(0, len(df), 10):
        time_stamp = df['time'][i]
        heroes = [hero_id_dict[hero] for hero in df['hero'][i:i+10]]
        features = df.iloc[i:i+10, [6,9,10,14,17,18,19]].values.flatten()
        team1_tower = df['rest_tower'][i]
        team1_barracks = df['rest_barracks'][i]
        team1_fort = df['fort'][i]
        team2_tower = df['rest_tower'][i+5]
        team2_barracks = df['rest_barracks'][i+5]
        team2_fort = df['fort'][i+5]
        label = [0, 0]
        if df['winner'][i] == df['team'][i]:
            label[0] = 1
        else:
            label[1] = 1

        features = np.concatenate((
            [time_stamp],
            heroes,
            features,
            [team1_tower, team1_barracks, team1_fort, team2_tower, team2_barracks, team2_fort]
        ))
        data.append((features, label))

    # 归一化处理，除了时间戳之外的部分
    scaler = MinMaxScaler()
    data_np = np.array([x[0][1:] for x in data])
    data_np_scaled = scaler.fit_transform(data_np)
    for i in range(len(data)):
        data[i] = (np.concatenate(([data[i][0][0]], data_np_scaled[i])), torch.tensor(data[i][1]))

    return data


# 自定义Dataset
class DotaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 超参数
train_val_split = 0.8
batch_size = 64
num_epochs = 20
learning_rate = 0.005
embedding_dim = 16
hidden_dim = 256
dropout_rate = 0.5  # 添加dropout率
l2_reg = 0.001

# 构建神经网络
class DotaNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout_rate):
        super(DotaNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(10 * embedding_dim + 76, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 2)  # 输出改为2
        self.softmax = nn.Softmax(dim=1)  # 激活函数改为Softmax

    def forward(self, x):
        heroes = x[:, 1:11].long()
        other_features = x[:, 11:]
        heroes_embedded = self.embedding(heroes).view(x.size(0), -1)
        x = torch.cat((heroes_embedded, other_features), dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.softmax(x)  # 激活函数改为Softmax
        return x



# 英雄ID字典
hero_id_dict = {'Abaddon': 0, 'AbyssalUnderlord': 1, 'Alchemist': 2, 'AncientApparition': 3, 'AntiMage': 4, 'ArcWarden': 5,
                'Axe': 6, 'Bane': 7, 'Batrider': 8, 'Beastmaster': 9, 'Bloodseeker': 10, 'BountyHunter': 11, 'Brewmaster': 12,
                'Bristleback': 13, 'Broodmother': 14, 'Centaur': 15, 'ChaosKnight': 16, 'Chen': 17, 'Clinkz': 18,
                'CrystalMaiden': 19, 'DarkSeer': 20, 'DarkWillow': 21, 'Dazzle': 22, 'DeathProphet': 23, 'Disruptor': 24,
                'DoomBringer': 25, 'DragonKnight': 26, 'DrowRanger': 27, 'Earthshaker': 28, 'EarthSpirit': 29, 'Elder': 30,
                'EmberSpirit': 31, 'Enchantress': 32, 'Enigma': 33, 'FacelessVoid': 34, 'Furion': 35, 'Grimstroke': 36,
                'Gyrocopter': 37, 'Huskar': 38, 'Invoker': 39, 'Jakiro': 40, 'Juggernaut': 41, 'KeeperOfTheLight': 42,
                'Kunkka': 43, 'Legion': 44, 'Leshrac': 45, 'Lich': 46, 'Life': 47, 'Lina': 48, 'Lion': 49, 'LoneDruid': 50,
                'Luna': 51, 'Lycan': 52, 'Magnataur': 53, 'Mars': 54, 'Medusa': 55, 'Meepo': 56, 'Mirana': 57, 'MonkeyKing': 58,
                'Morphling': 59, 'Naga': 60, 'Necrolyte': 61, 'Nevermore': 62, 'NightStalker': 63, 'Nyx': 64, 'Obsidian': 65,
                'Ogre': 66, 'Omniknight': 67, 'Oracle': 68, 'Pangolier': 69, 'PhantomAssassin': 70, 'PhantomLancer': 71,
                'Phoenix': 72, 'Puck': 73, 'Pudge': 74, 'Pugna': 75, 'QueenOfPain': 76, 'Rattletrap': 77, 'Razor': 78,
                'Riki': 79, 'Rubick': 80, 'SandKing': 81, 'Shadow': 82, 'ShadowShaman': 83, 'Shredder': 84, 'Silencer': 85,
                'SkeletonKing': 86, 'Skywrath': 87, 'Slardar': 88, 'Slark': 89, 'Snapfire': 90, 'Sniper': 91, 'Spectre': 92,
                'SpiritBreaker': 93, 'StormSpirit': 94, 'Sven': 95, 'Techies': 96, 'TemplarAssassin': 97, 'Terrorblade': 98,
                'Tidehunter': 99, 'Tinker': 100, 'Tiny': 101, 'Treant': 102, 'TrollWarlord': 103, 'Tusk': 104, 'Undying': 105,
                'Ursa': 106, 'VengefulSpirit': 107, 'Venomancer': 108, 'Viper': 109, 'Visage': 110, 'Void': 111, 'Warlock': 112,
                'Weaver': 113, 'Windrunner': 114, 'Winter': 115, 'Wisp': 116, 'WitchDoctor': 117, 'Zuus': 118}


class Dota2Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size, 1)
        self.beta = nn.Parameter(torch.tensor(1.0))  # 初始化beta为可学习参数

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = self.dropout(hidden)
        out = self.fc(hidden[-1])
        out = self._swish(out, self.beta)  # 使用带可学习beta的Swish激活函数
        return out.squeeze()

    def _swish(self, x, beta):
        return x * torch.sigmoid(beta * x)

def load_sample_data1(sample_path, sequence_length):
    df = pd.read_csv(sample_path)
    economy_diff = df['economy_diff'].values.reshape(-1, 1)
    winning_team = df['winning_team'].values[0]

    scaler = MinMaxScaler()
    features = scaler.fit_transform(economy_diff)

    X = [features[i:i+sequence_length] for i in range(len(features)-sequence_length)]
    y = [winning_team] * len(X)

    return np.array(list(zip(X, y)), dtype=object)

def load_sample_data2(sample_path, sequence_length):
    df = pd.read_csv(sample_path)
    economy_diff = df['experience_diff'].values.reshape(-1, 1)
    winning_team = df['winning_team'].values[0]

    scaler = MinMaxScaler()
    features = scaler.fit_transform(economy_diff)

    X = [features[i:i+sequence_length] for i in range(len(features)-sequence_length)]
    y = [winning_team] * len(X)

    return np.array(list(zip(X, y)), dtype=object)

def load_sample_data3(sample_path, sequence_length):
    df = pd.read_csv(sample_path)
    economy_diff = df['kda_diff'].values.reshape(-1, 1)
    winning_team = df['winning_team'].values[0]

    scaler = MinMaxScaler()
    features = scaler.fit_transform(economy_diff)

    X = [features[i:i+sequence_length] for i in range(len(features)-sequence_length)]
    y = [winning_team] * len(X)

    return np.array(list(zip(X, y)), dtype=object)


def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            outputs = model(inputs)
            predictions.append(outputs.item())
    return predictions





def predict_single_game(model1, model2, model3, sample_path, sequence_length, device, win_probabilities2):

    sample_data1 = load_sample_data1(sample_path, sequence_length)
    sample_data2 = load_sample_data2(sample_path, sequence_length)
    sample_data3 = load_sample_data3(sample_path, sequence_length)

    sample_dataset1 = Dota2Dataset(np.array([x for x, _ in sample_data1]), np.array([y for _, y in sample_data1]))
    sample_dataset2 = Dota2Dataset(np.array([x for x, _ in sample_data2]), np.array([y for _, y in sample_data2]))
    sample_dataset3 = Dota2Dataset(np.array([x for x, _ in sample_data3]), np.array([y for _, y in sample_data3]))

    sample_dataloader1 = DataLoader(sample_dataset1, batch_size=1, shuffle=False)
    sample_dataloader2 = DataLoader(sample_dataset2, batch_size=1, shuffle=False)
    sample_dataloader3 = DataLoader(sample_dataset3, batch_size=1, shuffle=False)

    predictions1 = evaluate(model1, sample_dataloader1, device)
    predictions2 = evaluate(model2, sample_dataloader2, device)
    predictions3 = evaluate(model3, sample_dataloader3, device)

    # 假设已经得到了这三个模型的胜率列表
    predictions1 = np.array(predictions1)
    predictions2 = np.array(predictions2)
    predictions3 = np.array(predictions3)

    # 计算胜率平均值并除以5
    min_length = min(len(predictions1), len(predictions2), len(predictions3))
    averaged_win_probs = np.zeros(min_length + 10)

    # 前10个点
    for i in range(10):
        averaged_win_probs[i] = win_probabilities2[i] / 5

    # 后面的点
    for i in range(10, min_length + 10):
        if i < len(win_probabilities2):
            averaged_win_probs[i] = (predictions1[i - 10] + predictions2[i - 10] + predictions3[i - 10] +
                                     win_probabilities2[i]) / 8
        else:
            averaged_win_probs[i] = (predictions1[i - 10] + predictions2[i - 10] + predictions3[i - 10]) / 6



    return predictions1, predictions2, predictions3, averaged_win_probs




def calculate_accuracy(predictions, actual_results, threshold=0):
    correct_predictions = 0
    actual_results_list = actual_results.tolist()

    for i, pred in enumerate(predictions):
        if (pred >= threshold and actual_results_list[i] == 1) or (pred < threshold and actual_results_list[i] == -1):
            correct_predictions += 1
    accuracy = correct_predictions / len(predictions)
    return accuracy



def main():
    folder_path = "H:/final_data/processed_data"
    folder_path2 = 'H:/final_data//base_data'
    sequence_length = 10
    input_size = 1
    hidden_size = 16
    num_layers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = LSTMModel(input_size, hidden_size, num_layers).to(device)
    model2 = LSTMModel(input_size, hidden_size, num_layers).to(device)
    model3 = LSTMModel(input_size, hidden_size, num_layers).to(device)

    model1.load_state_dict(torch.load("H:/systemcontribution/lstm_model_gold.pth"))  # 使用实际模型路径
    model2.load_state_dict(torch.load("H:/systemcontribution/lstm_model_XP.pth"))  # 使用实际模型路径
    model3.load_state_dict(torch.load("H:/systemcontribution/lstm_model_kda.pth"))  # 使用实际模型路径

    time_intervals = [(0, 120), (120,240), (240,360),(360,480),(480,600),(600,720),(720,840),(840,960),(960,1080),(1080,1200),(1200,1320),(1320,1440),(1440,1560),(1560,1680),(1680,1800),(1800,1920),(1920,2040),(2040,2160),(2160,2280) ,(2280,2400),(2400,2520), (2520,2640),(2640,2760),(2760,2880),(2880,3000),(3000, float('inf'))]


    accuracies1 = {interval: [] for interval in time_intervals}
    accuracies2 = {interval: [] for interval in time_intervals}
    accuracies3 = {interval: [] for interval in time_intervals}
    combined_accuracies = {interval: [] for interval in time_intervals}

    model = DotaNN(len(hero_id_dict), embedding_dim, hidden_dim, dropout_rate)
    model.load_state_dict(torch.load('H:/systemcontribution/FNN7.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # 将模型移动到相同的设备
    model.eval()  # 设置为评估模式

    total_combined_accuracies = []  ### 初始化一个空列表来收集所有游戏的平均胜率

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            sample_path = os.path.join(folder_path, file)
            sample_path2 = os.path.join(folder_path2, file)

            # 加载数据并计算 win_probabilities2
            preprocessed_data = preprocess_data(sample_path2)
            new_game_dataset = DotaDataset(preprocessed_data)
            new_game_dataloader = DataLoader(new_game_dataset, batch_size=1, shuffle=False)

            win_probabilities2 = []
            with torch.no_grad():
                for features, _ in new_game_dataloader:
                    features = features.clone().detach().float()
                    features = features.to(device)  # 将输入数据移动到相同的设备
                    probabilities = model(features)  # 返回的是胜率的概率
                    team1_win_probability = probabilities[0][0].item()
                    team2_win_probability = probabilities[0][1].item()
                    win_probabilities2.append(team1_win_probability - team2_win_probability)
            win_probabilities2 = np.array(win_probabilities2)

            preds1, preds2, preds3, avg_win_probs = predict_single_game(model1, model2, model3, sample_path,
                                                                        sequence_length, device, win_probabilities2)

            sample_df = pd.read_csv(sample_path)
            actual_results = sample_df['winning_team'][10:]  # 从第11个时间点开始
            actual_results_all = sample_df['winning_team']

            preds1 = pd.Series(preds1, index=actual_results.index)
            preds2 = pd.Series(preds2, index=actual_results.index)
            preds3 = pd.Series(preds3, index=actual_results.index)
            avg_win_probs = pd.Series(avg_win_probs, index=actual_results_all.index)

            for interval in time_intervals:
                start, end = interval

                actual_results_interval = actual_results[(sample_df['time'] >= start) & (sample_df['time'] < end)]

                if not actual_results_interval.empty:
                    preds1_interval = preds1[actual_results_interval.index].tolist()
                    preds2_interval = preds2[actual_results_interval.index].tolist()
                    preds3_interval = preds3[actual_results_interval.index].tolist()
                    avg_win_probs_interval = avg_win_probs[actual_results_interval.index].tolist()

                    acc1 = calculate_accuracy(preds1_interval, actual_results_interval)
                    acc2 = calculate_accuracy(preds2_interval, actual_results_interval)
                    acc3 = calculate_accuracy(preds3_interval, actual_results_interval)
                    combined_acc = calculate_accuracy(avg_win_probs_interval, actual_results_interval)

                    accuracies1[interval].append(acc1)
                    accuracies2[interval].append(acc2)
                    accuracies3[interval].append(acc3)
                    combined_accuracies[interval].append(combined_acc)

            ### 计算并收集当前游戏的全局平均胜率
            total_combined_accuracies.append(calculate_accuracy(avg_win_probs.tolist(), actual_results_all))

    ### 计算并输出所有游戏的全局平均胜率
    global_avg_combined_acc = sum(total_combined_accuracies) / len(total_combined_accuracies) if len(
        total_combined_accuracies) > 0 else 0
    print(f'Global Combined Model Average Accuracy: {global_avg_combined_acc:.6f}')



if __name__ == '__main__':
     main()


