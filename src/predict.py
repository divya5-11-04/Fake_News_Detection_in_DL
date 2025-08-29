import torch
import numpy as np
from transformers import AutoModel, BertTokenizerFast
from train import BERT_Arch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load model
bert = AutoModel.from_pretrained("bert-base-uncased")
model = BERT_Arch(bert)
model.load_state_dict(torch.load("../models/fake_news_model.pt", map_location=device))
model.to(device)
model.eval()

# ✅ Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
MAX_LENGTH = 15

def predict_news(news_list):
    encodings = tokenizer.batch_encode_plus(
        news_list, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    )
    seq, mask = encodings["input_ids"].to(device), encodings["attention_mask"].to(device)

    with torch.no_grad():
        preds = model(seq, mask).detach().cpu().numpy()
    preds = np.argmax(preds, axis=1)
    return ["Fake" if p == 1 else "True" for p in preds]

# ✅ Example usage
unseen_news = [
    "Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing",
    "Trump administration issues new rules on U.S. visa waivers"
]

print(predict_news(unseen_news))
