#📩SMS-Spam-ham-Detection-using-the-Machine-Learning-TF-IDF-


A **Machine Learning based SMS Spam Classifier** built using **Python, Scikit-learn, NLTK, and Streamlit**.  
The system uses **TF-IDF Vectorization** and a **Naive Bayes classifier** to predict whether an SMS message is **Spam** or **Ham (Legitimate)**.

---

# 🚀 Project Overview

Spam messages are a major issue in SMS communication.  
This project builds a **text classification model** that automatically detects spam messages.

The application provides:

- Real-time SMS prediction
- File upload for bulk prediction
- NLP based text preprocessing
- TF-IDF feature extraction
- Machine learning classification
- Downloadable prediction results

The system is deployed through a **Streamlit web interface** for easy interaction.

---

# 🧠 Machine Learning Workflow

```
SMS Dataset
     │
     ▼
Text Preprocessing
(Remove punctuation, lowercase, stopwords removal, lemmatization)
     │
     ▼
TF-IDF Vectorization
     │
     ▼
Naive Bayes Model Training
     │
     ▼
Model Serialization (Pickle)
     │
     ▼
Streamlit Web App
     │
     ▼
Spam / Ham Prediction
```

---

# 📂 Project Structure

```
SMS-Spam-Detection/
│
├── 7.py                      # Streamlit application
├── spam_model.pkl            # Trained Naive Bayes model
├── tfidf_vectorizer.pkl      # TF-IDF vectorizer
├── vectorizer.pkl            # Alternate vectorizer
├── SMSSpamCollection.txt     # Dataset
│
├── notebooks/
│   ├── 7.ipynb
│   └── 7a.ipynb
│
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

The project uses the **SMS Spam Collection Dataset**, which contains labeled SMS messages.

Example format:

```
ham   I'm gonna be home soon and I don't want to talk about this stuff anymore tonight
spam  WINNER!! You have been selected to receive a £900 prize reward!
```

Dataset Classes:

| Label | Meaning |
|------|------|
| ham | Legitimate message |
| spam | Unwanted promotional or fraudulent message |

---

# ⚙️ Technologies Used

| Technology | Purpose |
|-----------|--------|
| Python | Programming language |
| Scikit-learn | Machine learning algorithms |
| Streamlit | Web application interface |
| NLTK | Natural language processing |
| Pandas | Data processing |
| Pickle | Model serialization |

---

# 🔤 Text Preprocessing

Before prediction, the text is cleaned using NLP techniques.

Steps:

1. Remove special characters
2. Convert text to lowercase
3. Tokenization
4. Stopword removal
5. Lemmatization

Example:

Original message

```
WINNER!! You have won a prize
```

Processed message

```
winner prize
```

---

# 🧮 Feature Extraction

The model uses **TF-IDF (Term Frequency – Inverse Document Frequency)**.

TF-IDF converts text into numerical vectors representing word importance.

Example:

```
Message: "free prize now"

Vectorized Output:
[0.24, 0.51, 0.63, ...]
```

---

# 🤖 Machine Learning Model

Model used:

### Multinomial Naive Bayes

Why Naive Bayes?

- Works well for text classification
- Fast training and prediction
- High accuracy for spam detection
- Efficient for large text datasets

---

# 💻 Streamlit Web Application

The Streamlit app provides two main features.

### 1️⃣ Single Message Prediction

Users can manually enter a message to classify.

Example:

```
Input:
Congratulations! You have won a free iPhone
```

Output:

```
🚨 SPAM detected
Confidence: 0.98
```

---

### 2️⃣ Bulk Prediction via CSV Upload

Users can upload a CSV file containing multiple messages.

Example CSV format:

```
message
Hello how are you
You won a free lottery ticket
Meeting at 6pm
```

The app will:

- Predict spam or ham
- Display results in a table
- Allow downloading results as CSV

---

# 📦 Installation

Clone the repository:

```
git clone https://github.com/yourusername/sms-spam-detection.git
cd sms-spam-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit app:

```
streamlit run 7.py
```

Open the application in the browser:

```
http://localhost:8501
```

---

# 📈 Example Predictions

| Message | Prediction |
|-------|------|
| Hey are we meeting today? | Ham |
| You won a free vacation! Call now | Spam |
| Let's have lunch tomorrow | Ham |
| Claim your £1000 prize today | Spam |

---

# 📊 Future Improvements

Possible enhancements:

- Deep learning models (LSTM / BERT)
- Larger training dataset
- Email spam detection
- Model explainability
- Cloud deployment
- Mobile app integration

---

# 🧪 Applications

This system can be used in:

- Mobile messaging apps
- Telecom spam filtering
- Email spam detection
- Fraud detection systems

---

# 👨‍💻 Author

Machine Learning Project

---

# 📜 License

This project is open-source and available under the **MIT License**.
