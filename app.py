# python -m streamlit run app.py


import streamlit as st
import joblib

# Load models and vectorizers
sentiment_model = joblib.load("sentiment_model.pkl")
sentiment_tfidf = joblib.load("sentiment_tfidf.pkl")

urgency_model = joblib.load("urgency_model.pkl")
urgency_tfidf = joblib.load("urgency_tfidf.pkl")

# Priority weights
sentiment_weight = {
    'Negative': 2,
    'Neutral': 1,
    'Positive': 0
}

urgency_weight = {
    'High': 3,
    'Medium': 2,
    'Low': 1
}

def predict_sentiment(text):
    X = sentiment_tfidf.transform([text.lower()])
    return sentiment_model.predict(X)[0]

def predict_urgency(text):
    X = urgency_tfidf.transform([text.lower()])
    return urgency_model.predict(X)[0]

def compute_priority(text):
    sentiment = predict_sentiment(text)
    urgency = predict_urgency(text)
    score = sentiment_weight[sentiment] + urgency_weight[urgency]

    if score >= 4:
        final_priority = "Critical"
    elif score == 3:
        final_priority = "High"
    elif score == 2:
        final_priority = "Medium"
    else:
        final_priority = "Low"

    return sentiment, urgency, final_priority


# ---------------- UI ----------------

st.set_page_config(page_title="IT Ticket Priority Classifier", layout="centered")

st.title("🛠️ IT Support Ticket Priority Classifier")
st.write("Enter an IT support ticket to predict sentiment, urgency, and final priority.")

ticket_text = st.text_area(
    "Ticket Description",
    placeholder="e.g. Server is down and users cannot access the application"
)

if st.button("Analyze Ticket"):
    if ticket_text.strip() == "":
        st.warning("Please enter a ticket description.")
    else:
        sentiment, urgency, priority = compute_priority(ticket_text)

        st.subheader("Prediction Results")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Urgency:** {urgency}")
        st.write(f"**Final Priority:** {priority}")
