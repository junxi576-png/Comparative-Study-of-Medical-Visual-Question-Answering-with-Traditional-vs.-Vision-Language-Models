import torch
import torch.nn as nn
from torchvision import models

class MedVQA_Baseline(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_classes=2):
        super(MedVQA_Baseline, self).__init__()
        
        # 视觉编码器：ResNet-50 | Visual Encoder: ResNet-50
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        # 移除最后的分类层，保留特征提取部分 | Remove the final FC layer, keep feature extraction
        self.visual_encoder = nn.Sequential(*(list(resnet.children())[:-1]))
        
        # 文本编码器：LSTM | Text Encoder: LSTM
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        
        # 多模态融合与分类 | Multimodal Fusion and Classification
        # 输入维度 = ResNet(2048) + 双向LSTM(hidden_size * 2)
        # Input dim = ResNet(2048) + Bi-LSTM(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, questions):
        # 提取图像特征 | Extract image features
        # output shape: [batch_size, 2048]
        img_features = self.visual_encoder(images).view(images.size(0), -1)
        
        # 提取文本特征 | Extract text features
        embedded = self.embedding(questions)
        _, (h_n, _) = self.lstm(embedded)
        
        # 拼接双向LSTM的最后一个隐藏状态 | Concatenate last hidden states of Bi-LSTM
        # h_n shape: [num_layers * num_directions, batch, hidden_size]
        text_features = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        # 特征融合 | Multimodal Fusion
        combined = torch.cat((img_features, text_features), dim=1)
        
        # 分类输出 | Classification output
        return self.classifier(combined)