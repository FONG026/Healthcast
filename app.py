from flask import Flask, request, render_template
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import t
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

#prebuild a model using
stream = open("model.py")
read_file = stream.read()
exec(read_file)

# Declare a Flask app
app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":

        #Unpickle classifier
        regr = joblib.load("regr.pkl")
        #case:break
        # Get values through input bars

        var=request.form.get("choice")

        an = request.form.get("an")
        avgt = request.form.get("avgt")
        avgp = request.form.get("avgp")
        print(an)
        print(avgt)
        print(avgp)
        print(var)
       #Put inputs to dataframe
        X = pd.DataFrame([[an, avgt, avgp]], columns=['An', 'AvgT', 'pre'])

        #An,AvgT,pre,Ncas
        # Get prediction
        prediction = regr.predict(X)[0]

    else:
        prediction = ""

    return render_template("Home.html", output=prediction)
#'{:.0f}'.format(prediction)

# Running the app
if __name__ == '__main__':
    app.run(debug=True)

