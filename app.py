import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("mood_dataset.csv")
    df['mood_tomorrow'] = df['mood'].shift(-1)
    df = df.dropna(subset=['mood_tomorrow'])
    mood_map = {'sad': 0, 'neutral': 1, 'happy': 2}
    df['mood_tomorrow'] = df['mood_tomorrow'].map(mood_map)
    return df

df = load_data()

# Define feature columns
features = ['steps', 'distance_km', 'calories_burned', 'active_minutes',
            'sleep_hours', 'water_intake_liters']

# Extract features and target
X = df[features]
y = df['mood_tomorrow']

# Handle missing values if any
X = X.fillna(X.mean())
y = y.fillna(y.mode()[0])

# Train the model (only do this once, ideally save it)
try:
    model = joblib.load('mood_predictor_model.pkl')  # Load pre-trained model
except FileNotFoundError:
    model = RandomForestClassifier()
    model.fit(X, y)
    joblib.dump(model, 'mood_predictor_model.pkl')  # Save model

# Streamlit UI
st.title("🌤️ Predict Your Mood Tomorrow")

st.write("Enter today’s activity data to get your predicted mood for tomorrow:")

steps = st.number_input("Steps Walked", min_value=0, value=5000)
distance = st.number_input("Distance (km)", min_value=0.0, value=4.0)
calories = st.number_input("Calories Burned", min_value=0, value=300)
active_minutes = st.number_input("Active Minutes", min_value=0, value=30)
sleep = st.number_input("Sleep Hours", min_value=0.0, value=7.0)
water = st.number_input("Water Intake (liters)", min_value=0.0, value=2.0)

if st.button("Predict Mood"):
    input_data = np.array([[steps, distance, calories, active_minutes, sleep, water]])
    prediction = model.predict(input_data)[0]
    mood_label = {0: "😔 Sad", 1: "😐 Neutral", 2: "😊 Happy"}
    st.subheader(f"Predicted Mood for Tomorrow: {mood_label[prediction]}")

    # Show Feature Importance
    st.markdown("### Feature Importance")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})

    # Plotting with matplotlib for more control
    fig, ax = plt.subplots()
    ax.bar(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

st.markdown("---")
st.caption("Created with ❤️ using Streamlit")

# Handle missing values if any
X = X.fillna(X.mean())
y = y.fillna(y.mode()[0])

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🌤️ Predict Your Mood Tomorrow")

st.write("Enter today’s activity data to get your predicted mood for tomorrow:")

steps = st.number_input("Steps Walked", min_value=0, value=5000)
distance = st.number_input("Distance (km)", min_value=0.0, value=4.0)
calories = st.number_input("Calories Burned", min_value=0, value=300)
active_minutes = st.number_input("Active Minutes", min_value=0, value=30)
sleep = st.number_input("Sleep Hours", min_value=0.0, value=7.0)
water = st.number_input("Water Intake (liters)", min_value=0.0, value=2.0)

if st.button("Predict Mood"):
    input_data = np.array([[steps, distance, calories, active_minutes, sleep, water]])
    prediction = model.predict(input_data)[0]
    mood_label = {0: "😔 Sad", 1: "😐 Neutral", 2: "😊 Happy"}
    st.subheader(f"Predicted Mood for Tomorrow: {mood_label[prediction]}")

    # Show Feature Importance
    st.markdown("###Feature Importance")
    importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    st.bar_chart(importance_df.set_index("Feature"))

st.markdown("---")
st.caption("Created with  using Streamlit")
