import json
import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

def clean_text(text):
    """
    基础文本清洗：转换为小写并移除标点符号
    Basic text cleaning: convert to lowercase and remove punctuation
    """
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# 请确保文件名与你文件夹中的完全一致 | Ensure the filename matches your local folder
json_path = 'data/VQA_RAD_Dataset_Public.json' 

if not os.path.exists(json_path):
    print(f"错误：找不到文件 | Error: File not found: {json_path}")
else:
    # 1. 加载原始 JSON 数据 | Load raw JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 2. 转换为 DataFrame 并进行预处理 | Convert to DataFrame and preprocess
    df = pd.DataFrame(data)
    # 清洗问题和答案文本 | Clean question and answer text
    df['question_clean'] = df['question'].apply(clean_text)
    df['answer_clean'] = df['answer'].apply(clean_text)

    # 3. 划分数据集 (70% 训练, 15% 验证, 15% 测试) | Data Split (70% Train, 15% Val, 15% Test)
    # 首先划分出 70% 的训练集 | First, split out 70% for training
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    # 将剩下的 30% 对半划分为验证集和测试集 | Split remaining 30% equally into Val and Test
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 4. 保存为 CSV 文件以便后续训练使用 | Save to CSV for future training
    train_df.to_csv('data/train.csv', index=False)
    val_df.to_csv('data/val.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    print(f"数据处理成功！划分结果 | Data processing successful! Split results:")
    print(f"训练集 Train: {len(train_df)}, 验证集 Val: {len(val_df)}, 测试集 Test: {len(test_df)}")