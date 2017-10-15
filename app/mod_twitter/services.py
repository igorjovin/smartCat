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
from collections import defaultdict

from sklearn.cluster import KMeans, MiniBatchKMeans

import re
import logging
import sys
from time import time
 
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

    def __init__(self, tweet):
        self.tweet = tweet

    def perform_preprocessing(self):
        tweet_parts = self.tweet.split(" ")
        preprocessed_tweet = ""
        for part in tweet_parts:
            part.lower()
            part = self.removeRTSign(part)
            part = self.removeUsername(part)
            part = self.removeUrl(part)
            preprocessed_tweet += part + " "
        return preprocessed_tweet

    def removeUsername(self, part):
        if part.startswith('@'):
            part = ""
        return part

    def removeUrl(self, part):
        if part.startswith("http:") or part.startswith("https:"):
            part = ""
        return part

    def removeRTSign(self, part):
        if part == 'rt':
            part = ""
        return part


class TwitterKMeans():

    num_of_clusters = 5 #deafult

    def __init__(self, num_of_clusters):
        self.num_of_clusters = num_of_clusters

    def perform_clustering(self, tweets, num_of_clusters):
        km = KMeans(n_clusters=num_of_clusters, init='k-means++', max_iter=100, n_init=1,
                verbose=True)
        vectorizer = TfidfVectorizer(stop_words='english',
                                       norm='l2')
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
        #print(labels)


 
