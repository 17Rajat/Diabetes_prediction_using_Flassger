# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 11:34:11 2020

@author: Rajat
"""
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger
from flask import jsonify

app = Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
classifier = pickle.load(pickle_in)


@app.route('/',methods = ["Get"])
def predict_diabetes():
    
    """Lets find out whether a person is diabetic or not.
    ---
    
    parameters:
      - name : Pregnancies
        in : query
        type: number
        required : true
      - name : Glucose
        in : query
        type: number
        required : true
      - name : BloodPressure
        in : query
        type: number
        required : true
      - name : SkinThickness
        in : query
        type: number
        required : true
      - name : Insulin
        in : query
        type: number
        required : true
      - name : BMI
        in : query
        type: number
        required : true
      - name : DiabetesPedigreeFunction
        in : query
        type: number
        required : true
      - name : Age
        in : query
        type: number
        required : true
    responses:
        200:
            description : The output Values
    """
    Pregnancies = request.args.get("Pregnancies")
    Glucose = request.args.get("Glucose")
    BloodPressure = request.args.get("BloodPressure")
    SkinThickness = request.args.get("SkinThickness")
    Insulin = request.args.get("Insulin")
    BMI = request.args.get("BMI")
    DiabetesPedigreeFunction = request.args.get("DiabetesPedigreeFunction")
    Age = request.args.get("Age")
    prediction = classifier.predict([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    print(prediction)
    return "The answer is" + str(prediction)

@app.route('/predict_file',methods = ["POST"])
def predict_file():
    
    
    """Lets find out whether a person is diabetic or not.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
    """
      
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = classifier.predict(df_test)
    
    return str(list(prediction))


if __name__ == '__main__':
    app.run()

