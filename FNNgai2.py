import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# 数据预处理
def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    data = []
    for i in range(0, len(df), 10):
        time_stamp = df['time'][i]
        heroes = [hero_id_dict[hero] for hero in df['hero'][i:i+10]]
        features = df.iloc[i:i+10, [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values.flatten()
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
        self.fc1 = nn.Linear(10 * embedding_dim + 166, hidden_dim)
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

# 加载数据
data_path = 'H:/final_data/chulidata2'
data = []
for csv_file in os.listdir(data_path):
    data += preprocess_data(os.path.join(data_path, csv_file))

split_idx = int(len(data) * train_val_split)
train_data = data[:split_idx]
val_data = data[split_idx:]

train_dataset = DotaDataset(train_data)
val_dataset = DotaDataset(val_data)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DotaNN(len(hero_id_dict), embedding_dim, hidden_dim, dropout_rate).to(device)
criterion = nn.BCEWithLogitsLoss()  # 修改损失函数为多标签交叉熵损失函数
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)

time_points = [(0, 120), (120,240),(240,360),(360,480),(480,600),(600,720),(720,840),(840,960),(960,1080),(1080,1200),(1200,1320),(1320,1440),(1440,1560),(1560,1680),(1680,1800),(1800,1920),(1920,2040),(2040,2160),(2160,2280) ,(2280,2400),(2400,2520), (2520,2640),(2640,2760),(2760,2880),(2880,3000),(3000, float('inf'))]

accuracies = 0

for epoch in range(num_epochs):
    model.train()
    for i, (features, labels) in enumerate(train_dataloader):
        features = features.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    time_points_correct = [0] * len(time_points)
    time_points_total = [0] * len(time_points)
    with torch.no_grad():
        for features, labels in val_dataloader:
            features = features.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float).view(-1, 2)
            time_stamps = features[:, 0]

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = ((outputs > 0.5).float().cpu().numpy() > 0).astype(int)
            labels = labels.cpu().numpy().astype(int)

            correct += np.sum(np.all(preds == labels, axis=1))
            total += len(labels)

            for idx, (t_start, t_end) in enumerate(time_points):
                mask = (time_stamps >= t_start) & (time_stamps < t_end)
                mask = mask.cpu().numpy()

                tp_preds = preds[mask]
                tp_labels = labels[mask]

                tp_correct = np.sum(np.all(tp_preds == tp_labels, axis=1))
                tp_total = len(tp_labels)

                time_points_correct[idx] += tp_correct
                time_points_total[idx] += tp_total

    val_loss /= len(val_dataloader)
    accuracy = correct / total
    print(f'Epoch: {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Overall Accuracy: {accuracy:.4f}')
    for idx, (t_start, t_end) in enumerate(time_points):
        tp_correct = time_points_correct[idx]
        tp_total = time_points_total[idx]
        print(f'Accuracy for {t_start}s to {t_end}s: {tp_correct / tp_total:.4f}')

    accuracies = accuracies + accuracy
print(accuracies/20)





torch.save(model.state_dict(), 'H:/systemcontribution/FNN19.pth')







