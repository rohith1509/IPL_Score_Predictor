import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

import joblib

app = Flask(__name__) #Initialize the flask App
model=joblib.load('cricket_last.sav') # loading the trained model
sc=joblib.load('std_scaler.bin')

@app.route('/') # Homepage
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    features = [float(x) for x in request.form.values()]
    runs=int(features[0])
    wickets=int(features[1])
    overs=features[2]
    strike_scr=int(features[3])
    ns_scr=int(features[4])
    
    

    prediction = model.predict(sc.transform(np.array([[runs,wickets,overs,strike_scr,ns_scr]])))
    # making prediction


    return render_template('index.html', prediction_text='Predicted Score: {}'.format(int(prediction))) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=False)
