import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit Page Setup
st.set_page_config(page_title="Grievance Classifier", page_icon="📄")
st.title("🧠 Grievance Redressal AI System")
st.write("Enter a public complaint below. This AI will predict the appropriate department and urgency level.")

# Complaint Input
complaint = st.text_area("✍️ Complaint Text", placeholder="e.g. Water flooding near Sector 10 park due to pipe burst")

# Predict Button
if st.button("🔍 Classify"):
    if not complaint.strip():
        st.warning("Please enter a valid complaint.")
    else:
        # Predict department
        transformed_text = vectorizer.transform([complaint])
        department = model.predict(transformed_text)[0]
        st.success(f"🏛️ Predicted Department: **{department}**")

        # Enhanced urgency prediction (based on real-world keywords)
        complaint_lower = complaint.lower()
        if any(word in complaint_lower for word in ['urgent', 'immediately', 'serious', 'flood', 'fire', 'danger', 'collapsed', 'dead', 'electric shock']):
            urgency = "High"
        elif any(word in complaint_lower for word in ['soon', 'problem', 'issue', 'delay', 'not working', 'blocked']):
            urgency = "Medium"
        else:
            urgency = "Low"

        st.info(f"⚠️ Estimated Urgency: **{urgency}**")
