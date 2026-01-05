import streamlit as st
import pickle
import pandas as pd

# -------------------------
# Load model
# -------------------------
model_path = "naive.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("🚢 Titanic Survival Prediction (Naive Bayes Model)")
st.write("Enter passenger details to predict survival")

# --------------------------------
# USER INPUT FIELDS
# --------------------------------

passenger_id = st.number_input("Passenger ID", min_value=1, value=100)
p_class = st.selectbox("Passenger Class (p_class)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sib_sp = st.number_input("Siblings/Spouses Aboard (sib_sp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (parch)", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.0)
embarked = st.selectbox("Embarked", ["S", "C", "Q"])

# --------------------------------
# ENCODING MATCHING TRAINING
# --------------------------------

# Encode sex
sex_encoded = 1 if sex == "male" else 0

# Encode embarked
embarked_map = {"S": 0, "C": 1, "Q": 2}
embarked_encoded = embarked_map[embarked]

# --------------------------------
# BUILD INPUT DATAFRAME
# --------------------------------

input_data = pd.DataFrame({
    "passenger_id": [passenger_id],
    "p_class": [p_class],
    "sex": [sex_encoded],          # encoded
    "age": [age],
    "sib_sp": [sib_sp],
    "parch": [parch],
    "fare": [fare],
    "embarked": [embarked_encoded]  # encoded
})

# --------------------------------
# PREDICTION
# --------------------------------

if st.button("Predict"):
    try:
        pred = model.predict(input_data)[0]

        if pred == 1:
            st.success("✔ The passenger SURVIVED")
        else:
            st.error("❌ The passenger DID NOT SURVIVE")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
