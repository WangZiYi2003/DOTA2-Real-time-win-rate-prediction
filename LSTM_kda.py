import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


def load_data(path, seq_len):
    files = os.listdir(path)
    data = []
    for file in files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file))
            economy_diff = df['kda_diff'].values.reshape(-1, 1)
            winning_team = df['winning_team'].values[0]

            scaler = MinMaxScaler()
            features = scaler.fit_transform(economy_diff)

            X = [features[i:i+seq_len] for i in range(len(features)-seq_len)]
            y = [winning_team] * len(X)

            data.extend(list(zip(X, y)))

    return data

def train_test_split_data(data, test_size=0.2):
    X, y = zip(*data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

def load_sample_data(sample_path, sequence_length):
    df = pd.read_csv(sample_path)
    economy_diff = df['kda_diff'].values.reshape(-1, 1)
    winning_team = df['winning_team'].values[0]

    scaler = MinMaxScaler()
    features = scaler.fit_transform(economy_diff)

    X = [features[i:i+sequence_length] for i in range(len(features)-sequence_length)]
    y = [winning_team] * len(X)

    return np.array(list(zip(X, y)))

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().tolist())
    return predictions


def accuracy(predictions, labels):
    assert len(predictions) == len(labels), "Length of predictions and labels must be equal"
    correct = 0
    for pred, label in zip(predictions, labels):
        if (pred >= 0 and label == 1) or (pred < 0 and label == -1):
            correct += 1
    return correct / len(predictions)

def main():
    data_path = "H:/final_data/processed_data"
    model_save_path = "H:/systemcontribution/lstm_model_kda.pth"
    sample_path = "H:/final_data/processed_data/data1.csv"
    sequence_length = 10
    input_size = 1
    hidden_size = 16
    num_layers = 2
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 200

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data(data_path, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split_data(data)

    train_dataset = Dota2Dataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Dota2Dataset(X_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size, hidden_size, num_layers).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        test_predictions = evaluate(model, test_dataloader, device)
        test_accuracy = accuracy(test_predictions, y_test)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
        accuracies.append(test_accuracy)

    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')
    with open('H:/systemcontribution/accuracies.txt', 'a') as f:
        for acc in accuracies:
            f.write(str(acc) + ' ')
        f.write('\n')


if __name__ == '__main__':
    main()

