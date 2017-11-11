from __future__ import print_function
from tweepy import Stream
from tweepy.streaming import StreamListener

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import pandas as pd
import nltk
import pickle
import xgboost as xgb
import re
import logging
import sys
import numpy as np
from time import time
import os
from config import BASE_DIR, APP_STATIC

class TweetClassifier():

    classifier_name = ""
    classifier = MultinomialNB()#SVC()
    vectorizer = CountVectorizer()#TfidfVectorizer(min_df=2,
                             #max_df = 0.8,
                             #sublinear_tf=True)

    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        if classifier_name == "Naive Bayes":
            self.classifier = MultinomialNB()
        if classifier_name == "SVC":
            self.classifier = SVC(C = 30)
        if classifier_name == "SVR":
            self.classifier = SVR()
        if classifier_name == "Random Forest":
            self.classifier = RandomForestClassifier(n_estimators = 100, criterion='entropy')
        if classifier_name == "Linear Regression":
            self.classifier = LinearRegression()
        if classifier_name == "XGBoost":
            self.classifier = xgb.XGBClassifier(max_depth=10, n_estimators=500, learning_rate=0.05)

    def classify(self, hashtag):
        dataset = pd.read_csv(os.path.join(APP_STATIC, 'dataset-' + hashtag + '.csv'),
            header=None, names=['label','tweet'])
        y = dataset['label'].tolist()
        X = dataset['tweet'].tolist()
        X = self.vectorizer.fit_transform(X)
        #X = sp.hstack((X, sp.csr_matrix(np.ones((X.shape[0], 1)))))
        self.classifier.fit(X, y)
        self.save_classifier(hashtag)

    def predict(self, tweets, hashtag):
        loaded_classifier = self.load_classifier(hashtag)
        if not loaded_classifier:
            return None
        tweets_vectorized = self.vectorizer.transform(tweets)
        predictions = self.classifier.predict(tweets_vectorized)
        tweets_with_predictions = defaultdict(list)
        j = 0
        for i in predictions:
            tweets_with_predictions[i].append(tweets[j])
            j = j + 1
        return tweets_with_predictions

    def cross_validate(self, hashtag):
        dataset = pd.read_csv(os.path.join(APP_STATIC, 'dataset-' + hashtag + '.csv'),
            header=None, names=['label','tweet'])
        le = LabelEncoder()
        X = dataset['tweet'].tolist()
        y = dataset['label'].tolist()
        X = self.vectorizer.fit_transform(X)
        y = le.fit_transform(y)

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=7)
        self.classifier.fit(train_X, train_y)
        if self.is_regression():
            #y_pred = self.classifier.predict_proba(test_X)
            y_pred = self.classifier.predict(test_X)
            predictions = y_pred.argmax(axis=0)
        else:
            y_pred = self.classifier.predict(test_X)
            predictions = [round(value) for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(test_y, predictions)
        f1 = f1_score(test_y, y_pred, average='weighted')  
        return accuracy, f1

    def save_classifier(self, hashtag):
        classifier_file_name = "classifier-" + hashtag + "-" + self.classifier_name.replace(" ", "") + ".pickle"
        f = open(os.path.join(APP_STATIC, classifier_file_name), 'wb')
        pickle.dump(self.classifier, f)
        f.close()
        vectorizer_file_name = "vectorizer-" + hashtag + "-" + self.classifier_name.replace(" ", "") + ".pickle"
        f = open(os.path.join(APP_STATIC, vectorizer_file_name), 'wb')
        pickle.dump(self.vectorizer, f)
        f.close()

    def load_classifier(self, hashtag):
        classifier_file_name = "classifier-" + hashtag + "-" + self.classifier_name.replace(" ", "") + ".pickle"    
        vectorizer_file_name = "vectorizer-" + hashtag + "-" + self.classifier_name.replace(" ", "") + ".pickle"
        if os.path.isfile(os.path.join(APP_STATIC, classifier_file_name)):
            f = open(os.path.join(APP_STATIC, classifier_file_name), 'rb')
            self.classifier = pickle.load(f)
            f.close()
            f = open(os.path.join(APP_STATIC, vectorizer_file_name), 'rb')
            self.vectorizer = pickle.load(f)
            f.close()
            return True
        return False

    def is_regression(self):
        return self.classifier_name == "Linear Regression" or self.classifier_name == "SVR"



 
