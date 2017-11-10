from __future__ import print_function
from tweepy import Stream
from tweepy.streaming import StreamListener

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import pandas as pd
import nltk
import pickle
import xgboost as xgb
import re
import logging
import sys
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
        if classifier_name == "SVM":
            self.classifier = SVC()
        if classifier_name == "Random Forest":
            self.classifier = RandomForestClassifier(n_estimators = 30)
        if classifier_name == "Linear Regression":
            self.classifier = LinearRegression()
        if classifier_name == "XGBoost":
            self.classifier = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

    def classify(self, hashtag):
        dataset = pd.read_csv(os.path.join(APP_STATIC, 'dataset-' + hashtag + '.csv'),
            header=None, names=['label','tweet'])
        print(dataset.shape)
        y = dataset['label'].tolist()
        X = dataset['tweet'].tolist()
        X = self.vectorizer.fit_transform(X)
        X = sp.hstack((X, sp.csr_matrix(np.ones((X.shape[0], 1)))))
        self.classifier.fit(X, y)
        self.save_classifier(hashtag)

    def predict(self, tweets, hashtag):
        self.load_classifier(hashtag)
        tweets_vectorized = self.vectorizer.transform(tweets)
        predictions = self.classifier.predict(tweets_vectorized)
        print(predictions)
        tweets_with_predictions = defaultdict(list)
        j = 0
        for i in predictions:
            print("Prediction %s" % i)
            print(tweets[j])
            tweets_with_predictions[i].append(tweets[j])
            j = j + 1
        return tweets_with_predictions

    def save_classifier(self, hashtag):
        classifier_file_name = "classifier-" + hashtag + "-" + self.classifier_name.replace(" ", "") + ".pickle"
        f = open(os.path.join(APP_STATIC, classifier_file_name), 'wb')
        pickle.dump(self.classifier, f)
        f.close()
        contains_classifier = False
        with open(os.path.join(APP_STATIC, "classifiers-used.txt"), 'rwb') as f:
            for line in f:
                if classifier_name == line:
                    contains_classifier = True
                    break
            if not contains_classifier:
                f.write(classifier_name + "\n")
        f.close()

    def load_classifier(self, hashtag):
        classifier_file_name = "classifier-" + hashtag + "-" + self.classifier_name.replace(" ", "") + ".pickle"    
        f = open(os.path.join(APP_STATIC, classifier_file_name), 'rb')
        self.classifier = pickle.load(f)
        f.close()



 
