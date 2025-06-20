from flask import Flask,request,render_template
import joblib
import pandas as pd
import numpy as np

app=Flask(__name__)


# Loading the model, scaler and label encoder
model=joblib.load('career.pkl')
scaler=joblib.load('scaler.pkl')
le=joblib.load('label_encoder.pkl')

@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/career',methods=['GET'])
def career_form():
    return render_template('career.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        # Get data from the users
        data={
            'Linguistic':float(request.form['Linguistic']),
            'Musical':float(request.form['Musical']),
            'Bodily':float(request.form['Bodily']),
            'Logical - Mathematical':float(request.form['Logical - Mathematical']),
            'Spatial-Visualization':float(request.form['Spatial-Visualization']),
            'Interpersonal':float(request.form['Interpersonal']),
            'Intrapersonal':float(request.form['Intrapersonal']),
            'Naturalist':float(request.form['Naturalist'])
        }

        # Validate the inputs
        for key,value in data.items():
            if not (0<=value<=20):
                return render_template('career.html',error=f"{key} score must be between 0 and 20")

        # Create a DataFrame for the  inputs
        df=pd.DataFrame([data])

        # Normalize the numerical features
        numerical_cols=['Linguistic','Musical','Bodily','Logical - Mathematical','Spatial-Visualization','Interpersonal','Intrapersonal','Naturalist']
        df[numerical_cols]=scaler.transform(df[numerical_cols])

        # Add dummy categorical columns
        rating_columns=['P1','P2','P3','P4','P5','P6','P7','P8']
        for col in rating_columns:
            df[col]=1  # Default to AVG; can be enhanced with form inputs

        # Predict
        prediction=model.predict(df)
        career=le.inverse_transform(prediction)[0]

        return render_template('result.html',career=career)

    except ValueError as e:
        return render_template('career.html',error="Please enter valid numerical values for all scores")
    except Exception as e:
        return render_template('career.html',error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)