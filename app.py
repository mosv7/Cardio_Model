import numpy as np 
import pickle 
import streamlit as st 


loaded_model = pickle.load(open("diabetes_model.pkl", "rb"))

def diabetes_predictions(input_data):
    input_as_numpy = np.array(input_data)
    input_reshaped = input_as_numpy.reshape(1,-1)

    preds = loaded_model.predict(input_reshaped)
    if preds[0] == 0:
        return 'the person does not do cardio'
    else:
        return "the person does cardio"
    
    


def main():
    st.title("Diabetes prediction")
    age = st.text_input('age')
    height = st.text_input('height')
    weight = st.text_input('weight')
    ap_hi = st.text_input('ap_hi')
    ap_lo = st.text_input('ap_lo')
    cholesterol = st.text_input('cholesterol')
    gluc = st.text_input('gluc')
    smoke = st.text_input('smoke')
    alco = st.text_input('alco')
    active = st.text_input('active')
  

    diagnoses = ''

    if  st.button("Test result"):
        diagnoses = diabetes_predictions([age,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active])

        st.success(diagnoses)


if __name__ == "__main__":
    main()