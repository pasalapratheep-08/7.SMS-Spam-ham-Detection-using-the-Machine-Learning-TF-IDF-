import streamlit as st
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

st.title("📩 SMS Spam/Ham Predictor")
st.write("This app uses **TF-IDF Vectorization** and a **Naive Bayes model** to classify SMS messages.")

# -------------------------------
# 2. Download NLTK Resources
# -------------------------------
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

load_nltk()

# -------------------------------
# 3. Load ML Assets
# -------------------------------
@st.cache_resource
def load_assets():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)

    return tfidf, model


try:
    tfidf, model = load_assets()
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please run the training script first.")
    st.stop()

# -------------------------------
# 4. Text Preprocessing
# -------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.lower().split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------
# 5. Prediction Function
# -------------------------------
def predict_message(message):
    cleaned = preprocess_text(message)
    vectorized = tfidf.transform([cleaned]).toarray()

    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized).max()

    return prediction, probability, cleaned, vectorized

# -------------------------------
# 6. User Input Section
# -------------------------------
st.subheader("✉️ Enter a Message")

user_input = st.text_area("Type SMS content here:", height=150)

if st.button("🔍 Predict Message"):
    if user_input.strip():

        prediction, prob, cleaned, vectorized = predict_message(user_input)

        if prediction == "spam":
            st.error(f"🚨 SPAM detected (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ HAM (Legitimate message) (Confidence: {prob:.2f})")

        with st.expander("🔎 TF-IDF Details"):
            st.write("Cleaned Text:", cleaned)
            st.write("TF-IDF Non-zero values:", vectorized[vectorized > 0])

    else:
        st.warning("Please enter a message.")

# -------------------------------
# 7. File Upload (Browse Option)
# -------------------------------
st.divider()
st.subheader("📂 Test Multiple Messages (Browse File)")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing messages",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "message" not in df.columns:
        st.error("CSV must contain a column named **message**")
    else:

        results = []

        for msg in df["message"]:
            prediction, prob, _, _ = predict_message(str(msg))

            results.append({
                "Message": msg,
                "Prediction": prediction,
                "Confidence": round(prob, 2)
            })

        result_df = pd.DataFrame(results)

        st.success("Prediction Complete ✅")
        st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Results",
            csv,
            "spam_predictions.csv",
            "text/csv"
        )