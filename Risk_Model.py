import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('Risk_model1.pkl')

st.title("Healthcare Risk Stratification App")

age = st.number_input("Age", min_value=0, value=25)
length_of_stay = st.number_input(
    "Length of Stay (days)",
    min_value=0,
    value=1
)

treatment_cost = st.number_input(
    "Treatment Cost",
    min_value=0.0,
    value=1000.0
)

if st.button("Predict"):

    input_data = pd.DataFrame(
        [[age, treatment_cost, 0]],
        columns=['Age', 'TreatmentCost', 'AbnormalLabCount']
    )

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk Patient")
    else:
        st.success("Low Risk Patient")

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
        st.write(
            f"Risk Probability: {probability*100:.2f}%"
        )
