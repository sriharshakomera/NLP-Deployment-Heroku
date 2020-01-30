# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 23:38:22 2019

@author: Sriharsha Komera
"""

### Imprting my dataset
import pandas as pd
import pickle

path= 'F:\\Krish\\NLP\\Spam Classifier\\SpamClassifier-master\\smsspamcollection\\SMSSpamCollection'

messages=pd.read_csv(path, sep='\t', names=["label","message"])

### Data Cleaning and pre-processing
import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
ps=PorterStemmer()
lm=WordNetLemmatizer()

corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-z]',' ',messages['message'] [i])
    review=review.lower()
    review=review.split()
    
    ##review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=[lm.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)

###Creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(max_features=5000)
X=cv.fit_transform(corpus)
X = cv.fit_transform(corpus) # Fit the Data   
pickle.dump(cv, open('F:\\Krish\\NLP\\Spam Classifier\\tranform.pkl', 'wb'))


y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values

###train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.20, random_state=0)

###Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model=MultinomialNB().fit(X_train,y_train)

y_pred= spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print(cr)
accuracy=accuracy_score(y_test,y_pred)

pickle.dump(spam_detect_model,open('F:\\Krish\\NLP\\Spam Classifier\\NLP_Model.pkl','wb'))

NLP_Model=pickle.load(open('F:\\Krish\\NLP\\Spam Classifier\\NLP_Model.pkl','rb'))
#	df= pd.read_csv("spam.csv", encoding="latin-1")
#	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#	# Features and Labels
#	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
#	X = df['message']
#	y = df['label']
#	
#	# Extract Feature With CountVectorizer
#	cv = CountVectorizer()
#	X = cv.fit_transform(X) # Fit the Data
#    
#    pickle.dump(cv, open('tranform.pkl', 'wb'))
#    
#    
#	from sklearn.model_selection import train_test_split
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#	#Naive Bayes Classifier
#	from sklearn.naive_bayes import MultinomialNB
#
#	clf = MultinomialNB()
#	clf.fit(X_train,y_train)
#	clf.score(X_test,y_test)
#    filename = 'nlp_model.pkl'
#    pickle.dump(clf, open(filename, 'wb'))
    
	#Alternative Usage of Saved Model
	# joblib.dump(clf, 'NB_spam_model.pkl')
	# NB_spam_model = open('NB_spam_model.pkl','rb')
	# clf = joblib.load(NB_spam_model)
