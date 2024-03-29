# -*- coding: utf-8 -*-
"""Pim3.0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oo5oAxKNxeJavJjuTZjW334--ysgBLFP
"""

import pandas as pd

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import linear_model
import joblib
from lightgbm import LGBMRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import t
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error
pd.set_option('display.float_format', lambda x: '%.9f' % x)

"""Importing panda"""

df=pd.read_csv("./data/nigeria.csv",encoding='ISO-8859-1')

"""importing the dataset"""

X = df[['An','AvgT', 'pre']]
y = df['Ncas']
X = X.values

"""y=ax+b assigning our data set column 
to two created variables
"""


"""importing sklearn the library for ml"""

regr = linear_model.LinearRegression()
regr.fit(X, y)
joblib.dump(regr, "regr.pkl")


"""instancing the linear regeression 
object
the fit() help us to create our model th
based on our two variables X and y
"""

##predictedcase = regr.predict([[29.06,80]])
##print(predictedcase)
#df.pre=df.pre*12
#df.info()

"""Analysing the data"""

#regr.predict([[2027,29,500000]]) #make predictions





