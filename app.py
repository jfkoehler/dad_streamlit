import streamlit as st 
import numpy as np
import pickle

sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input('Sepal Width (cm)')
petal_length = st.number_input('Petal Length')
petal_width = st.number_input('Petal Width')

new_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

with open('rf.pkl', 'rb') as f:
    model = pickle.load(f)
    
prediction = model.predict(new_flower)
st.write(prediction)