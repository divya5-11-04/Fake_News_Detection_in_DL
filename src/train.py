import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, BertTokenizerFast
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import create_data_loader

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load dataset
true_data = pd.read_csv('../data/a1_True.csv')
fake_data = pd.read_csv('../data/a2_Fake.csv')

true_data['Target'] = ['True'] * len(true_data)
fake_data['Target'] = ['Fake'] * len(fake_data)

data = pd.concat([true_data, fake_data]).sample(frac=1).reset_index(drop=True)
data['label'] = pd.get_dummies(data.Target)['Fake']

# ✅ Train-Validation-Test Split
train_text, temp_text, train_labels, temp_labels = train_test_split(
    data['title'], data['label'], test_size=0.3, stratify=data['Target'], random_state=2018
)
val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text, temp_labels, test_size=0.5, stratify=temp_labels, random_state=2018
)

# ✅ Load pretrained BERT
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

MAX_LENGTH = 15
def encode(texts):
    return tokenizer.batch_encode_plus(
        texts.tolist(), max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    )

train_encodings = encode(train_text)
val_encodings = encode(val_text)

train_y = torch.tensor(train_labels.tolist()).long()
val_y = torch.tensor(val_labels.tolist()).long()

train_loader = create_data_loader(train_encodings['input_ids'], train_encodings['attention_mask'], train_y, sampler_type="random")
val_loader = create_data_loader(val_encodings['input_ids'], val_encodings['attention_mask'], val_y, sampler_type="sequential")

# ✅ Freeze BERT layers
for param in bert.parameters():
    param.requires_grad = False

# ✅ Model Definition
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

model = BERT_Arch(bert).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.NLLLoss()

# ✅ Training Loop
def train_epoch():
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        sent_id, mask, labels = [r.to(device) for r in batch]
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = criterion(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return total_loss / len(train_loader)

def evaluate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            sent_id, mask, labels = [r.to(device) for r in batch]
            preds = model(sent_id, mask)
            loss = criterion(preds, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# ✅ Train model
best_valid_loss = float("inf")
epochs = 2

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss = train_epoch()
    val_loss = evaluate()

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        torch.save(model.state_dict(), "../models/fake_news_model.pt")

    print(f"Training Loss: {train_loss:.3f} | Validation Loss: {val_loss:.3f}")
