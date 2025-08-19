import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the saved model and transformers
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

st.title('Insurance Premium Price Prediction Dashboard')
st.write("This dashboard predicts the insurance premium price based on various health parameters.")

st.subheader("Enter Your Health Details")

# Create input forms for the features
age = st.slider('Age', 18, 65, 30)
height = st.number_input('Height (cm)', 140, 200, 170)
weight = st.number_input('Weight (kg)', 40, 130, 70)
diabetes = st.selectbox('Diabetes', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
blood_pressure_problems = st.selectbox('Blood Pressure Problems', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
any_transplants = st.selectbox('Any Transplants', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
any_chronic_diseases = st.selectbox('Any Chronic Diseases', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
history_of_cancer_in_family = st.selectbox('History of Cancer in Family', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
number_of_major_surgeries = st.number_input('Number of Major Surgeries', 0, 3, 0)
known_allergies = st.selectbox('Known Allergies', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

# Predict button
if st.button('Predict Premium Price'):
    # Create a dictionary with the input data
    data = {
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Diabetes': diabetes,
        'BloodPressureProblems': blood_pressure_problems,
        'AnyTransplants': any_transplants,
        'AnyChronicDiseases': any_chronic_diseases,
        'KnownAllergies': known_allergies,
        'HistoryOfCancerInFamily': history_of_cancer_in_family,
        'NumberOfMajorSurgeries': number_of_major_surgeries
    }

    input_df = pd.DataFrame([data])

    # Feature Engineering
    input_df['BMI'] = input_df['Weight'] / ((input_df['Height'] / 100) ** 2)
    risk_cols = ['Diabetes', 'BloodPressureProblems', 'AnyTransplants','AnyChronicDiseases', 'HistoryOfCancerInFamily','NumberOfMajorSurgeries']
    input_df['HealthRiskScore'] = input_df[risk_cols].sum(axis=1)
    
    with st.sidebar:
        st.header("📊 Health Indicators")
        st.metric("BMI", f"{input_df['BMI'].iloc[0]:.2f}")
        st.metric("Health Risk Score", f"{input_df['HealthRiskScore'].iloc[0]}")

    # Create AgeGroup feature
    input_df['AgeGroup'] = pd.cut(input_df['Age'],
                                bins=[0, 30, 45, 60, 100],
                                labels=['Young', 'Middle-aged', 'Senior', 'Elderly'],
                                right=True)

    # Select the numerical and categorical features for scaling and encoding
    num_features = ['Age', 'Height', 'Weight', 'BMI','NumberOfMajorSurgeries', 'HealthRiskScore']
    cat_feature = ['AgeGroup']

    # Scale numerical features
    input_df[num_features] = scaler.transform(input_df[num_features])

    # One-hot encode the AgeGroup feature
    encoded_age_group = one_hot_encoder.transform(input_df[cat_feature])
    encoded_age_group_df = pd.DataFrame(
        encoded_age_group,
        columns=one_hot_encoder.get_feature_names_out(cat_feature),
        index=input_df.index
    )

    # Combine all features for prediction

    final_input = pd.concat([input_df.drop(columns=cat_feature), encoded_age_group_df], axis=1)

    # Reorder columns to match the training data used by the model
    expected_features = ['Age', 'Height', 'Weight', 'Diabetes', 'BloodPressureProblems',
                        'AnyTransplants', 'AnyChronicDiseases', 'HistoryOfCancerInFamily',
                        'NumberOfMajorSurgeries','BMI','HealthRiskScore','AgeGroup_Middle-aged','AgeGroup_Senior','AgeGroup_Young']

    final_input = final_input[expected_features]

    # Predict the premium price
    predicted_price = rf_model.predict(final_input)[0]

    all_tree_preds = np.stack([tree.predict(final_input) for tree in rf_model.estimators_], axis=1)
    prediction = all_tree_preds.mean(axis=1)[0]
    lower = np.percentile(all_tree_preds, 2.5, axis=1)[0]
    upper = np.percentile(all_tree_preds, 97.5, axis=1)[0]
    interval = (lower, upper)

    with st.sidebar:
        st.header(" Premium Prediction")
        if prediction is not None:
            st.metric("Predicted Premium", f"${prediction:,.2f}")
            st.write(f"Model Prediction Range: **${interval[0]:,.2f} - ${interval[1]:,.2f}**")
        else:
            st.write("Click **Predict Premium** to see results.")
