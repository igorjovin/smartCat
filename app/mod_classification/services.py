from __future__ import print_function
import tweepy
from config import BASE_DIR, APP_STATIC, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
from tweepy import OAuthHandler
import app.mod_twitter.services as twitter_service

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
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
import logging
from time import time
import os, glob
from config import BASE_DIR, APP_STATIC

class TweetClassifier():

    classifier_name = ""
    classifier = OneVsRestClassifier(MultinomialNB())
    vectorizer = CountVectorizer()#TfidfVectorizer(min_df=2,
                             #max_df = 0.8,
                             #sublinear_tf=True)

    def __init__(self, classifier_name):
        self.classifier_name = classifier_name
        if classifier_name == "Naive Bayes":
            self.classifier = OneVsRestClassifier(MultinomialNB())
        if classifier_name == "SVM (SVC)":
            self.classifier = OneVsRestClassifier(LinearSVC(C = 30, multi_class="crammer_singer"))
        if classifier_name == "Random Forest":
            self.classifier = RandomForestClassifier(n_estimators = 150, criterion='entropy')
        if classifier_name == "Logistic Regression":
            self.classifier = OneVsRestClassifier(LogisticRegression(C = 30, max_iter = 1000))
        if classifier_name == "XGBoost":
            self.classifier = OneVsRestClassifier(xgb.XGBClassifier(max_depth=10, n_estimators=500, learning_rate=0.05))

    def classify(self, hashtag):
        dataset = pd.read_csv(os.path.join(APP_STATIC, 'dataset-' + hashtag + '.csv'),
            header=None, names=['label','tweet'])
        y = dataset['label'].tolist()
        for i, label in enumerate(y):
            if ";" in label:
                y[i] = [ single_label.strip() for single_label in label.split(";") if single_label.strip() ]
        class_list = self.load_classes(hashtag)
        multi_label_binarizer = MultiLabelBinarizer(classes=list(class_list))
        y = multi_label_binarizer.fit_transform(y)
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
        tweets_with_predictions = defaultdict(list)
        indexes_with_preprocessed_tweets = dict()
        indexes_with_original_tweets = dict()
        classes = self.load_classes(hashtag)
        j = 0
        for i in predictions:
            contains_1 = False
            for index, single_prediction in enumerate(i):
                if single_prediction == 1:
                    contains_1 = True
                    tweets_with_predictions[j].append(classes[index])
            if contains_1:
                indexes_with_preprocessed_tweets[j] = preprocessed_tweets[j]
                indexes_with_original_tweets[j] = original_tweets[j]
                j = j + 1
        return classes, tweets_with_predictions, indexes_with_preprocessed_tweets, indexes_with_original_tweets

    def cross_validate(self, hashtag):
        dataset = pd.read_csv(os.path.join(APP_STATIC, 'dataset-' + hashtag + '.csv'),
            header=None, names=['label','tweet'])
        X = dataset['tweet'].tolist()
        y = dataset['label'].tolist()
        X = self.vectorizer.fit_transform(X)
        for i, label in enumerate(y):
            if ";" in label:
                y[i] = [ single_label.strip() for single_label in label.split(";") if single_label.strip() ]
        class_list = self.load_classes(hashtag)
        multi_label_binarizer = MultiLabelBinarizer(classes=list(class_list))
        y = multi_label_binarizer.fit_transform(y)

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=7)
        self.classifier.fit(train_X, train_y)
        y_pred = self.classifier.predict(test_X)
        predictions = [value for value in y_pred]
        # evaluate predictions
        accuracy = accuracy_score(test_y, y_pred, normalize=True)#self.hamming_score(test_y, y_pred)accuracy_score(test_y, y_pred, normalize=False)
        f1 = f1_score(test_y, y_pred, average='weighted')  
        return accuracy, f1

    def hamming_score(self, y_true, y_pred, normalize=True, sample_weight=None):
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0] )
            set_pred = set(np.where(y_pred[i])[0] )
            tmp_a = None
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred))/\
                        float( len(set_true.union(set_pred)) )
            acc_list.append(tmp_a)
        return np.max(acc_list)

    def classifier_exists(self, hashtag):
        classifier_file_name = "classifier-" + hashtag + "-" + self.classifier_name.replace(" ", "") + ".pickle"    
        if os.path.isfile(os.path.join(APP_STATIC, classifier_file_name)):
            try:
                with open(os.path.join(APP_STATIC, classifier_file_name), 'rb') as f:
                    return True
            except IOError:
                logging.warning('Unable to write to load classifier/vectorizer with name %s.' % classifier_file_name)
                return False
        logging.warning("No classifier with name %s " % classifier_file_name)
        return False

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
            try:
                with open(os.path.join(APP_STATIC, classifier_file_name), 'rb') as f:
                    self.classifier = pickle.load(f)
                    f.close()
            except IOError:
                return False
            try: 
                with open(os.path.join(APP_STATIC, vectorizer_file_name), 'rb') as f:
                    self.vectorizer = pickle.load(f)
                    f.close()
            except IOError:
                logging.warning('Unable to write to load classifier/vectorizer with name %s.' % hashtag + "-" + self.classifier_name.replace(" ", ""))
                return False   
            return True
        logging.warning("No classifier with name %s " % hashtag + "-" + self.classifier_name.replace(" ", ""))
        return False

    def write_cluster_tweets_to_dataset_csv(self, hashtag, class_names, clustered_tweets):
        try:
            csv_file_name = self.get_dataset_csv_file_path(hashtag)
            with open(os.path.join(APP_STATIC, csv_file_name), 'a') as f:
                cw = csv.writer(f)
                indexes_with_tweets = dict()
                indexes_with_clusters = defaultdict(list)
                all_tweets = list()
                i = 0
                for label, value in clustered_tweets.iteritems(): #for each tweet write a tuple of classes it's in
                    for item in value:
                        if item in all_tweets:
                            ind = all_tweets.index(item)
                            indexes_with_clusters[ind].append(label)
                        else:
                            all_tweets.append(item)
                            indexes_with_tweets[i] = item
                            indexes_with_clusters[i].append(label)
                            i += 1

                for index, val in indexes_with_tweets.iteritems():
                    label = ""
                    for cl in indexes_with_clusters[index]:
                        label += str(class_names[cl]) + ";"
                    cw.writerow([str(label), str(val.encode('utf-8'))])
                f.close()
                return True
        except IOError:
            logging.warning('Unable to write to CSV dataset.')
            return False

    def write_to_dataset_csv(self, hashtag, indexes_with_tweets, tweets_with_predictions):
        tweets_with_predictions_to_append, indexes_with_tweets_to_append = self.get_tweets_to_append(hashtag, indexes_with_tweets, tweets_with_predictions)
        csv_file_name = self.get_dataset_csv_file_path(hashtag)
        try:
            with open(self.get_dataset_csv_file_path(hashtag), "a") as f:
                cw = csv.writer(f)
                for index, value in indexes_with_tweets_to_append.iteritems():
                    label = ""
                    for cl in tweets_with_predictions_to_append[index]:
                        label += cl + ";"
                    cw.writerow([str(label), str(value.encode('utf-8'))])
                f.close()
        except IOError:
            logging.warning('No CSV file with name %s .' % csv_file_name)

    def load_classes(self, hashtag):
        class_list = list()
        dataset = pd.read_csv(os.path.join(APP_STATIC, 'dataset-' + hashtag + '.csv'),
            header=None, names=['label','tweet'])
        y = dataset['label'].tolist()
        for i, label in enumerate(y):
            if ";" in label:
                for single_label in label.split(";"):
                    if single_label.strip() not in class_list and single_label.strip() != "":
                        class_list.append(single_label.strip())
            else:
                if label.strip() not in class_list and label.strip() != "":
                    class_list.append(label.strip())
        return class_list

    def get_tweets_to_append(self, hashtag, indexes_with_tweets, tweets_with_predictions):
        file_path = self.get_dataset_csv_file_path(hashtag)
        existing_tweets = pd.read_csv(file_path, header=None, names=["label", "tweet"])
        existing_tweet_list = existing_tweets["tweet"].tolist()
        new_indexes_with_tweets = dict()
        new_tweets_with_predictions = dict()
        i = 0
        for index, value in indexes_with_tweets.iteritems():
            if value not in existing_tweet_list:
                if tweets_with_predictions[index]: #if there are any tags associated with the tweet
                    new_indexes_with_tweets[i] = value
                    new_tweets_with_predictions[i] = tweets_with_predictions[index]
                    i += 1
        return new_tweets_with_predictions, new_indexes_with_tweets    

    def get_dataset_csv_file_path(self, hashtag):
        file_name = 'dataset-' + hashtag + '.csv'
        return os.path.join(os.path.join(APP_STATIC, file_name))

    def is_regression(self):
        return self.classifier_name == "Linear Regression" or self.classifier_name == "SVR"

classifier = TweetClassifier("Naive Bayes")
tweets_with_predictions = defaultdict(list) #key = index of tweet; value = list of classes
indexes_with_preprocessed_tweets = dict() #key = index of tweet; value = tweet text
indexes_with_original_tweets = dict()
indexes_with_tweets_to_show = dict()
class_names = {}
classes = set()
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
    delete_dataset_csv(cluster_hashtag)
    delete_all_classifiers(cluster_hashtag)
    classifier.write_cluster_tweets_to_dataset_csv(cluster_hashtag, class_names, preprocessed_tweets_with_clusters)
    classifier.classify(cluster_hashtag)

def retrain_classifier(hashtag, classifier_name):
    global classifier
    global preprocessed_tweets_with_predictions
    classifier = TweetClassifier(classifier_name)
    #classifier.write_to_dataset_csv(hashtag, preprocessed_tweets_with_predictions)
    classifier.classify(hashtag)

def predict(hashtag, classifier_name, items_per_page, filter_classes): #ovo treba da bude sakupljanje tvitova
    global classifier
    global tweets_with_predictions
    global indexes_with_original_tweets
    global indexes_with_preprocessed_tweets
    global classes
    global showing_original_tweets
    showing_original_tweets = True
    tweets_with_predictions = defaultdict(list)
    indexes_with_original_tweets = dict()
    indexes_with_preprocessed_tweets = dict()
    preprocessed_tweets = list()
    original_tweets = list()
    classifier = TweetClassifier(classifier_name)
    if not classifier.classifier_exists(hashtag):
        return {}, {}, {}
    preprocessed_tweets, original_tweets = twitter_service.get_tweets_from_api(hashtag, items_per_page)
    classes, tweets_with_predictions, indexes_with_preprocessed_tweets, indexes_with_original_tweets = classifier.predict(preprocessed_tweets, original_tweets, hashtag)
    if filter_classes is not None and filter_classes:
        tweets_with_predictions, indexes_with_preprocessed_tweets, indexes_with_original_tweets = get_tweets_from_filter_classes(filter_classes, tweets_with_predictions, indexes_with_preprocessed_tweets, indexes_with_original_tweets)
    indexes_with_tweets_to_show = indexes_with_original_tweets
    return classes, tweets_with_predictions, indexes_with_tweets_to_show#get_tweets_to_show()

def cross_validate(hashtag, classifier_name):
    hashtag = twitter_service.process_hashtag(hashtag)
    classifier = TweetClassifier(classifier_name)
    accuracy, f1 = classifier.cross_validate(hashtag)
    accuracy = round(accuracy * 100.0, 2)
    f1 = round(f1 * 100.0, 2)
    return accuracy, f1

def add_tag_to_tweet(tweet_index, desired_tag):
    global tweets_with_predictions
    global classes
    if desired_tag not in classes:
        classes.append(desired_tag)
    if desired_tag not in tweets_with_predictions[tweet_index]:
        tweets_with_predictions[tweet_index].append(desired_tag)
    indexes_with_tweets_to_show = indexes_with_preprocessed_tweets
    if showing_original_tweets:
        indexes_with_tweets_to_show = indexes_with_original_tweets
    return classes, tweets_with_predictions, indexes_with_tweets_to_show

def remove_tag_from_tweet(tweet_index, tag):
    global tweets_with_predictions
    global classes
    tweets_with_predictions[tweet_index].remove(tag)
    indexes_with_tweets_to_show = indexes_with_preprocessed_tweets
    if showing_original_tweets:
        indexes_with_tweets_to_show = indexes_with_original_tweets
    return classes, tweets_with_predictions, indexes_with_tweets_to_show

def create_new_class(desired_name):
    global classes
    classes.add(desired_name)

def save_to_dataset(hashtag, classifier_name):
    global classifier
    global tweets_with_predictions
    global indexes_with_preprocessed_tweets
    classifier = TweetClassifier(classifier_name)
    classifier.write_to_dataset_csv(hashtag, indexes_with_preprocessed_tweets, tweets_with_predictions)

def switch_tweet_view():
    global indexes_with_preprocessed_tweets
    global indexes_with_original_tweets
    global showing_original_tweets
    if showing_original_tweets:
        showing_original_tweets = False
        indexes_with_tweets_to_show = indexes_with_preprocessed_tweets
    else:
        showing_original_tweets = True
        indexes_with_tweets_to_show = indexes_with_original_tweets
    return classes, tweets_with_predictions, indexes_with_tweets_to_show

def get_tweets_from_filter_classes(filter_classes, tweets_with_predictions, indexes_with_preprocessed_tweets, indexes_with_original_tweets):
    tweets_with_predictions_return = dict()
    indexes_with_preprocessed_tweets_return = dict()
    indexes_with_original_tweets_return = dict()
    for tweet_index, tweet_classes in tweets_with_predictions.iteritems():
        belongs_to_class = True
        for filter_cl in filter_classes:
            if filter_cl not in tweet_classes:
                belongs_to_class = False
                break
        if belongs_to_class:
            tweets_with_predictions_return[tweet_index] = tweet_classes
            indexes_with_preprocessed_tweets_return[tweet_index] = indexes_with_preprocessed_tweets[tweet_index]
            indexes_with_original_tweets_return[tweet_index] = indexes_with_original_tweets[tweet_index]
    return tweets_with_predictions_return, indexes_with_preprocessed_tweets_return, indexes_with_original_tweets_return

def get_classes_from_dataset(hashtag):
    class_list = list()
    dataset = pd.read_csv(os.path.join(APP_STATIC, 'dataset-' + hashtag + '.csv'),
        header=None, names=['label','tweet'])
    y = dataset['label'].tolist()
    for i, label in enumerate(y):
        if ";" in label:
            for single_label in label.split(";"):
                if single_label.strip() not in class_list and single_label.strip() != "":
                    class_list.append(single_label.strip())
        else:
            if label.strip() not in class_list and label.strip() != "":
                class_list.append(label.strip())
    return class_list

def delete_dataset_csv(hashtag):
    try:
        csv_file_name = get_dataset_csv_file_path(hashtag)
        os.remove(csv_file_name)
        return True
    except IOError:
        logging.warning('Unable to delete CSV dataset.')
        return False

def delete_all_classifiers(hashtag):
    for filename in glob.glob(os.path.join(APP_STATIC, "classifier-" + hashtag + "*")):
        os.remove(filename) 
    for filename in glob.glob(os.path.join(APP_STATIC, "vectorizer-" + hashtag + "*")):
        os.remove(filename) 

def get_dataset_csv_file_path(hashtag):
    file_name = 'dataset-' + hashtag + '.csv'
    return os.path.join(os.path.join(APP_STATIC, file_name))


 
