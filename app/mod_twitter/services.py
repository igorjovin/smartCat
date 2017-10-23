from __future__ import print_function
from tweepy import Stream
from tweepy.streaming import StreamListener

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.svm import SVC
from collections import defaultdict
import pandas as pd
import nltk
nltk.download('wordnet')
from nltk import WordNetLemmatizer

from sklearn.cluster import KMeans, MiniBatchKMeans

import re
import logging
import sys
from time import time
import os
from config import BASE_DIR, APP_STATIC
 
class TwitterStreamListener(StreamListener):
 
    def on_data(self, data):
        try:
            with open('python.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True
 
    def on_error(self, status):
        print(status)
        return True

class TweetPreprocessor():

    tweet = ""
    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)

    def __init__(self, tweet):
        self.tweet = tweet
        reload(sys)
        sys.setdefaultencoding('utf-8')

    def perform_preprocessing(self):
        #ls = LancasterStemmer()
        lemmatizer = WordNetLemmatizer()
        self.emoji_pattern.sub(r'', self.tweet)
        tweet_parts = self.tweet.split(" ")
        preprocessed_tweet = ""
        for part in tweet_parts:
            #part = ls.stem(part)
            part = part.decode('utf-8')
            part = part.lower()
            part = self.removeStopWords(part)
            part = self.removeUsername(part)
            part = self.removeUrl(part)
            part = self.removePunctuation(part)
            part = self.removeNumbers(part)
            part = lemmatizer.lemmatize(part)
            preprocessed_tweet += part + " "
        return preprocessed_tweet

    def removeUsername(self, part):
        if part.startswith('@'):
            part = ""
        return part

    def removeUrl(self, part):
        part = re.sub(r'^https?:\/\/.*[\r\n]*', "", part, flags=re.MULTILINE)
        return part

    def removePunctuation(self, part):
        part = re.sub("[^\\w\\s]", "", part)
        return part

    def removeNumbers(self, part):
        part = re.sub('[0-9]+', "", part)
        return part

    def removeStopWords(self, part):
        with open(os.path.join(APP_STATIC, 'stopwords.txt')) as f:
            lines = f.readlines()
        for line in lines:
            if line in part or part == line:
                part = ""
        return part


class TwitterKMeans():

    num_of_clusters = 5 #default

    def __init__(self, num_of_clusters):
        self.num_of_clusters = num_of_clusters

    def perform_clustering(self, tweets, num_of_clusters):
        km = KMeans(n_clusters=num_of_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=True)
        vectorizer = TfidfVectorizer(stop_words='english',
                                       norm='l2',
                                       lowercase=True)
        X = vectorizer.fit_transform(tweets)
        print("Clustering sparse data with %s" % km)
        t0 = time()
        km.fit_transform(X)
        print("done in %0.3fs" % (time() - t0))
        print()
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        labels = km.predict(X)
        tweets_with_groups = defaultdict(list)
        j = 0
        for i in labels:
            print("Label %s" % i)
            print(tweets[j])
            tweets_with_groups[i].append(tweets[j])
            j = j + 1
        return tweets_with_groups

class TweetClassifier():

    classifier = SVC()
    vectorizer = TfidfVectorizer(min_df=2,
                             max_df = 0.8,
                             sublinear_tf=True)

    def classify(self):
        dataset = pd.read_csv(os.path.join(APP_STATIC, 'dataset.csv'),
            header=None, names=['label','tweet'])
        print(dataset.shape)
        y = dataset['label'].tolist()
        X = dataset['tweet'].tolist()
        X = self.vectorizer.fit_transform(X)
        self.classifier.fit(X, y)

    def predict(self, tweets):
        tweets = self.vectorizer.transform(tweets)
        predictions = self.classifier.predict(tweets)
        print(predictions)
        return predictions



 
