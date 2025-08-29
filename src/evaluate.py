import torch
import numpy as np
from sklearn.metrics import classification_report
from transformers import AutoModel, BertTokenizerFast
from train import BERT_Arch, encode, test_text, test_labels  # reuse from train.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert = AutoModel.from_pretrained("bert-base-uncased")
model = BERT_Arch(bert)
model.load_state_dict(torch.load("../models/fake_news_model.pt", map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
test_encodings = encode(test_text)
test_seq, test_mask = test_encodings['input_ids'].to(device), test_encodings['attention_mask'].to(device)
test_y = torch.tensor(test_labels.tolist()).long().to(device)

with torch.no_grad():
    preds = model(test_seq, test_mask).detach().cpu().numpy()

preds = np.argmax(preds, axis=1)
print(classification_report(test_y.cpu(), preds))
