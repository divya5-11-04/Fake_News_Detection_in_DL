# 📰 Fake News Detection using BERT

Detect fake vs real news headlines with **state-of-the-art NLP (BERT)**.  
Built with HuggingFace Transformers, PyTorch, and PyCaret.

## 📌 Features
- Pretrained BERT model fine-tuned for binary classification
- 70:15:15 Train-Validation-Test split
- Class balancing visualization
- Model performance metrics (Precision, Recall, F1-score)
- Predictions on unseen real-world news

## 📊 Dataset
- **True News:** `a1_True.csv`  
- **Fake News:** `a2_Fake.csv`  

*(Dataset not uploaded due to size/licensing; use your own.)*

## ⚙️ Tech Stack
- Python, PyTorch, HuggingFace Transformers
- Scikit-learn, Matplotlib
- PyCaret for quick ML experimentation

## 🚀 Training
```bash
python src/train.py

📈 Results

Accuracy: XX%

F1-score: YY%

Example predictions:

✅ "Trump administration issues new rules on U.S. visa waivers" → True

❌ "Donald Trump Sends Out Embarrassing New Year’s Eve Message" → Fake

🎯 Future Improvements

Use RoBERTa / DistilBERT for comparison

Deploy with Streamlit/Gradio

Add multilingual support

👩‍💻 Author

Divya Monga – LinkedIn
 | GitHub