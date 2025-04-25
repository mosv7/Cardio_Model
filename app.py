import numpy as np 
import pickle 
import streamlit as st 


loaded_model = pickle.load(open("cardio_model.pkl", "rb"))

def cardio_predictions(input_data):
    input_as_numpy = np.array(input_data)
    input_reshaped = input_as_numpy.reshape(1,-1)

    preds = loaded_model.predict(input_reshaped)
    if preds[0] == 0:
        return 'Low cardiovascular risk'
    else:
        return "High cardiovascular risk"
    
    

def main():
    st.title("Cardiovascular Disease Prediction")
    age = st.text_input('Age')
    height = st.text_input('Height (cm)')
    weight = st.text_input('Weight (kg)')
    ap_hi = st.text_input('Systolic BP')
    ap_lo = st.text_input('Diastolic BP')
    cholesterol = st.text_input('Cholesterol level')
    gluc = st.text_input('Glucose level')
    smoke = st.text_input('Smoker (1/0)')
    alco = st.text_input('Alcohol intake (1/0)')
    active = st.text_input('Physical activity (1/0)')

    diagnoses = ''

    if st.button("Check Cardiovascular Risk"):
        diagnoses = cardio_predictions([age, height, weight, ap_hi, ap_lo, 
                                      cholesterol, gluc, smoke, alco, active])
        st.success(diagnoses)


if __name__ == "__main__":
    main()