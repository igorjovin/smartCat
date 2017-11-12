from __future__ import print_function
import tweepy
from config import BASE_DIR, APP_STATIC, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import app.mod_twitter.services as twitter_service

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
import csv
import numpy as np
from time import time
import os
from config import BASE_DIR, APP_STATIC

class TweetClassifier():

    classifier_name = ""
    classifier = MultinomialNB()
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

        self.classifier.fit(X, y)
        self.save_classifier(hashtag)
        return True

    def predict(self, preprocessed_tweets, original_tweets, hashtag):
        loaded_classifier = self.load_classifier(hashtag)
        if not loaded_classifier:
            return None
        tweets_vectorized = self.vectorizer.transform(preprocessed_tweets)
        predictions = self.classifier.predict(tweets_vectorized)
        preprocessed_tweets_with_predictions = defaultdict(list)
        original_tweets_with_predictions = defaultdict(list)
        j = 0
        for i in predictions:
            preprocessed_tweets_with_predictions[i].append(preprocessed_tweets[j])
            original_tweets_with_predictions[i].append(original_tweets[j])
            j = j + 1
        return preprocessed_tweets_with_predictions, original_tweets_with_predictions

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

    def write_cluster_tweets_to_dataset_csv(self, class_names, clustered_tweets):
        f = open(self.get_dataset_csv_file_path(hashtag), "a")
        cw = csv.writer(f)
        for label, value in clustered_tweets.iteritems():
            for item in value:
                cw.writerow([str(class_names[label]), str(item.encode('utf-8'))])
        f.close()

    def write_to_dataset_csv(self, hashtag, tweets_with_predictions):
        tweets_to_append = self.get_tweets_to_append(hashtag, tweets_with_predictions)
        f = open(self.get_dataset_csv_file_path(hashtag), "a")
        cw = csv.writer(f)
        for label, value in tweets_to_append.iteritems():
            for item in value:
                cw.writerow([str(label), str(item.encode('utf-8'))])
        f.close()

    def get_tweets_to_append(self, hashtag, tweets_with_predictions):
        file_path = self.get_dataset_csv_file_path(hashtag)
        existing_tweets = pd.read_csv(file_path, header=None, names=["label", "tweet"])
        existing_tweet_list = existing_tweets["tweet"].tolist()
        new_tweets_with_predictions = defaultdict(list)
        for label, value in tweets_with_predictions.iteritems():
            for item in value:
                if item not in existing_tweet_list:
                    new_tweets_with_predictions[label].append(item)
        return new_tweets_with_predictions    

    def get_dataset_csv_file_path(self, hashtag):
        file_name = 'dataset-' + hashtag + '.csv'
        return os.path.join(os.path.join(APP_STATIC, file_name))

    def is_regression(self):
        return self.classifier_name == "Linear Regression" or self.classifier_name == "SVR"

classifier = TweetClassifier("Naive Bayes")
original_tweets_with_predictions = defaultdict(list)
preprocessed_tweets_with_predictions = defaultdict(list)
class_names = {}
class_list = list()
hashtag = ""
showing_original_tweets = True

auth = OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)#, proxy="http://proxy.uns.ac.rs:8080")

def select_classifier(classifier_name):
    global classifier
    classifier = TweetClassifier(classifier_name)

def train_classifier(preprocessed_tweets_with_clusters, cluster_names, cluster_hashtag):
    global class_names
    global classifier
    class_names = cluster_names
    classifier.write_cluster_tweets_to_dataset_csv(class_names, preprocessed_tweets_with_clusters)
    classifier.classify(cluster_hashtag)

def retrain_classifier(hashtag, classifier_name):
    global classifier
    global preprocessed_tweets_with_predictions
    classifier = TweetClassifier(classifier_name)
    classifier.write_to_dataset_csv(hashtag, preprocessed_tweets_with_predictions)
    classifier.classify(hashtag)

def predict(hashtag, classifier_name):
    global classifier
    global preprocessed_tweets_with_predictions
    global class_list
    global showing_original_tweets
    showing_original_tweets = True
    preprocessed_tweets_with_predictions = defaultdict(list)
    original_tweets_with_predictions = defaultdict(list)
    preprocessed_tweets = list()
    original_tweets = list()
    hashtag = twitter_service.process_hashtag(hashtag)
    for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en").items(100):
        tweet_text = twitter_service.process(tweet, hashtag)
        if tweet_text not in preprocessed_tweets:
           preprocessed_tweets.append(tweet_text)
           original_tweets.append(tweet.text)

    classifier = TweetClassifier(classifier_name)
    preprocessed_tweets_with_predictions, original_tweets_with_predictions = classifier.predict(preprocessed_tweets, original_tweets, hashtag)
    if preprocessed_tweets_with_predictions is None:
        message = "There is no classifier " + classifier_name
        print(message)
    else:
        class_list = list(preprocessed_tweets_with_predictions.keys())
    return preprocessed_tweets_with_predictions#get_tweets_to_show()

def cross_validate(hashtag, classifier_name):
    hashtag = twitter_service.process_hashtag(hashtag)
    classifier = TweetClassifier(classifier_name)
    accuracy, f1 = classifier.cross_validate(hashtag)
    accuracy = round(accuracy * 100.0, 2)
    f1 = round(f1 * 100.0, 2)
    return accuracy, f1

def move_tweet_to_class(key, desired_key, tweet_index):
    global preprocessed_tweets_with_predictions
    global class_list
    if desired_key not in class_list:
        class_list.append(desired_key)
    tweet = preprocessed_tweets_with_predictions[key][tweet_index]
    del preprocessed_tweets_with_predictions[key][tweet_index]
    preprocessed_tweets_with_predictions[desired_key].append(tweet)
    return preprocessed_tweets_with_predictions#get_tweets_to_show()

def remove_tweet_from_class(key, tweet_index):
    global preprocessed_tweets_with_predictions
    global class_names
    tweet = preprocessed_tweets_with_predictions[key][tweet_index]
    del preprocessed_tweets_with_predictions[key][tweet_index]
    return preprocessed_tweets_with_predictions#get_tweets_to_show()

def create_new_class(desired_name):
    global class_names
    class_names[len(class_names)] = desired_name

def switch_tweet_view():
    global preprocessed_tweets_with_predictions
    global original_tweets_with_predictions
    global showing_original_tweets
    if showing_original_tweets:
        showing_original_tweets = False
        return preprocessed_tweets_with_predictions
    else:
        showing_original_tweets = True
        return original_tweets_with_predictions
    return tweets_to_show

def get_tweets_to_show():
    global preprocessed_tweets_with_predictions
    global original_tweets_with_predictions
    global showing_original_tweets
    if showing_original_tweets:
        return original_tweets_with_predictions
    else:
        return preprocessed_tweets_with_predictions




 
