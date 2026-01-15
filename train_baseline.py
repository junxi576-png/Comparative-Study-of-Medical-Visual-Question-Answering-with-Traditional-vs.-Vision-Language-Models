import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from baseline_model import MedVQA_Baseline # 导入你之前的模型类 | Import your previous model class

# 1. 配置路径与参数 | Configuration Paths and Parameters
IMG_DIR = "data/VQA_RAD Image Folder"
TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# 2. 构建简单的词汇表 | Build a Simple Vocabulary
def build_vocab(df):
    """ 将文本转换为词典映射 | Map text to a dictionary """
    all_text = " ".join(df['question_clean'].tolist())
    words = sorted(list(set(all_text.split())))
    # 0 预留给 Padding | 0 is reserved for Padding
    vocab = {word: i+1 for i, word in enumerate(words)}
    vocab['<PAD>'] = 0
    return vocab

# 3. 数据集类 | Dataset Class
class VQARADDataset(Dataset):
    def __init__(self, csv_file, vocab, transform=None):
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.transform = transform
        # 将答案转换为分类标签 | Map unique answers to category IDs
        self.ans_to_id = {ans: i for i, ans in enumerate(self.df['answer_clean'].unique())}
        self.num_classes = len(self.ans_to_id)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载图像并转换 | Load and transform image
        img_path = os.path.join(IMG_DIR, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 处理问题文本：转换为 ID 并进行截断/填充 | Process text: convert to IDs and pad/truncate
        tokens = [self.vocab.get(w, 0) for w in row['question_clean'].split()]
        max_len = 20
        if len(tokens) < max_len: 
            tokens += [0] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        
        label = self.ans_to_id.get(row['answer_clean'], 0)
        return image, torch.tensor(tokens), torch.tensor(label), row['answer_type']

# 4. 训练准备 | Training Preparation
# 图像预处理：缩放、张量化、归一化 | Image Preprocessing: Resize, ToTensor, Normalize
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_df = pd.read_csv(TRAIN_CSV)
vocab = build_vocab(train_df)
train_dataset = VQARADDataset(TRAIN_CSV, vocab, transform)
val_dataset = VQARADDataset(VAL_CSV, vocab, transform)

# 初始化模型、加载器、损失函数和优化器 | Initialize Model, Loaders, Loss, and Optimizer
model = MedVQA_Baseline(vocab_size=len(vocab), num_classes=train_dataset.num_classes).to(DEVICE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. 训练循环 | Training Loop
print(f"开始在 {DEVICE} 上训练基准模型... | Starting training on {DEVICE}...")



for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, qs, labels, _ in train_loader:
        imgs, qs, labels = imgs.to(DEVICE), qs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(imgs, qs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # 验证环节 | Validation Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, qs, labels, _ in val_loader:
            imgs, qs, labels = imgs.to(DEVICE), qs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs, qs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {100 * correct / total:.2f}%")

# 保存模型权重 | Save model weights
torch.save(model.state_dict(), "baseline_best_model.pth")
print("基准模型训练完成并已保存权重。 | Baseline training complete. Weights saved.")