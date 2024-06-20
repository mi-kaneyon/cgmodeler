import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm

# カスタムデータセットの定義
class PoseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        joint_data_path = os.path.join(self.data_dir, self.data_files[idx])
        with open(joint_data_path, 'r') as f:
            joint_data = json.load(f)
        mask_sum = torch.tensor(joint_data["mask_sum"], dtype=torch.float32).unsqueeze(0)
        mask_sum = mask_sum / 1e6  # 正規化
        return mask_sum

# 簡単な3Dモデル生成用のNN
class Simple3DModel(nn.Module):
    def __init__(self):
        super(Simple3DModel, self).__init__()
        self.fc = nn.Linear(1, 3)  # 単純な例として1次元を3次元に変換

    def forward(self, x):
        return self.fc(x)

# データのロード
script_dir = os.path.dirname(__file__)
data_dir = os.path.join(script_dir, '../data')
dataset = PoseDataset(data_dir)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# モデルの初期化
model = Simple3DModel().to(device)

# 損失関数とオプティマイザの設定
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# モデルの学習
num_epochs = 50  # エポック数を増やしてファインチューニング
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, torch.zeros_like(outputs))  # 仮のターゲット
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}")

# モデル保存用ディレクトリの作成
model_dir = os.path.join(script_dir, '../models')
os.makedirs(model_dir, exist_ok=True)

# モデルの保存
model_path = os.path.join(model_dir, '3d_model.pth')
torch.save(model.state_dict(), model_path)
