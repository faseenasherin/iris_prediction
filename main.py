import streamlit as st
import pandas as pd
from os import path
import numpy as np
import pickle

st.title("Flower species predictor")
petal_length=st.number_input("please choose a petal length",
                             placeholder="Enter value ranges between 1.0 and 6.9",min_value=1.0,max_value=6.9,value=None)
petal_width=st.number_input("please choose a petal width",
                             placeholder="Enter value ranges between 0.1 and 2.5",min_value=0.1,max_value=2.5,value=None)
sepal_length=st.number_input("please choose a sepal length",
                             placeholder="Enter value ranges between 4.3 and 7.9",min_value=4.3,max_value=7.9,value=None)
sepal_width=st.number_input("please choose a sepal width",
                            placeholder="Enter value ranges between 2.0 and 4.4",min_value=2.0,max_value=4.4,value=None)
#prepare the dataframe for prediction
df_user_input=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
                         columns=['sepal_length','sepal_width','petal_length','petal_width'])

#using the .pkl file creating an ML model named 'iris_predictor'
model_path=path.join("Model","iris_classifier.pkl")
with open (model_path,'rb') as file:
    iris_predictor=pickle.load(file)
st.write(df_user_input)

dict_species={0:'setosa',1:'versicolor',2:'viginica'}

if st.button("Predict Species"):
    if((petal_length==None)or(petal_width==None)
       or(sepal_length==None)or(sepal_width==None)):
        #will be executed when any of the values is not entered properly
        st.write("Please fill all values")
    else:
        #prediction can be done here
        perdicted_species=iris_predictor.predict(df_user_input)
        #perdicted_species[0] will give us the value inth dataframe
        #we use that value to find the corresponding species from the dictionary 'dict_species'
        st.write("the Species is",dict_species[perdicted_species[0]])