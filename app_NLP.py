# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:25:01 2020

@author: Sriharsha Komera
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

# load the model from disk
filename = 'F:\\Krish\\NLP\\Spam Classifier\\NLP_Model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('F:\\Krish\\NLP\\Spam Classifier\\tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)