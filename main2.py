import pandas as pd  
import numpy as np  
import pickle  
import streamlit as st  
from PIL import Image  
  
# Load the trained model  
pickle_in1 = open('classifier2.pkl', 'rb')  
classifier1 = pickle.load(pickle_in1)  
  
def prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1):  
    prediction = classifier1.predict([[sepal_length1, sepal_width1, petal_length1, petal_width1]])  
    return prediction  
  
def main():  
    st.title("Iris Flower Prediction")  
  
    html_temp = """  
    <div style="background-color: #FFFF00; padding: 16px">  
    <h1 style="color: #000000; text-align: center;">Streamlit Iris Flower Classifier ML App</h1>  
    </div>  
    """  
  
    st.markdown(html_temp, unsafe_allow_html=True)  
  
    sepal_length1 = st.text_input("Sepal Length", "Type Here")  
    sepal_width1 = st.text_input("Sepal Width", "Type Here")  
    petal_length1 = st.text_input("Petal Length", "Type Here")  
    petal_width1 = st.text_input("Petal Width", "Type Here")  
    result = ""  
  
    if st.button("Predict"):  
        result = prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1)  
    st.write('The output of the above is', result)  
  
if __name__ == '__main__':  
    main()  

