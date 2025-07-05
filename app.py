
import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit App
st.set_page_config(page_title="Grievance Classifier", page_icon="🗂️")
st.title("🧠 Grievance Department Classifier")
st.write("Enter a citizen complaint and this AI will classify the responsible department and urgency.")

# Input
complaint = st.text_area("✍️ Complaint Text", placeholder="e.g., Water leakage near my house in Sector 5")

if st.button("Classify"):
    if complaint.strip() == "":
        st.warning("Please enter a complaint.")
    else:
        # Preprocess + Predict
        transformed = vectorizer.transform([complaint])
        prediction = model.predict(transformed)[0]

        # Display results
        st.success(f"✅ Assigned Department: **{prediction}**")

        # Optionally, predict urgency (basic heuristic)
        if any(word in complaint.lower() for word in ['urgent', 'immediately', 'serious', 'danger']):
            urgency = "High"
        elif any(word in complaint.lower() for word in ['soon', 'problem', 'issue']):
            urgency = "Medium"
        else:
            urgency = "Low"

        st.info(f"📶 Estimated Urgency Level: **{urgency}**")
