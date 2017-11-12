from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from collections import defaultdict
import nltk

from sklearn.cluster import KMeans, MiniBatchKMeans

import re
import logging
import sys
from time import time
import os
from tweepy import OAuthHandler
from tweepy import Stream
from collections import defaultdict
from config import BASE_DIR, APP_STATIC, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
import app.mod_twitter.services as twitter_service

class TwitterKMeans():

    num_of_clusters = 5 #default

    def __init__(self, num_of_clusters):
        self.num_of_clusters = num_of_clusters

    def perform_clustering(self, preprocessed_tweets, original_tweets, num_of_clusters):
        km = KMeans(n_clusters=num_of_clusters, init='k-means++', max_iter=200, n_init=15,
                verbose=True)
        vectorizer = TfidfVectorizer(stop_words='english',
                                       norm='l2', ngram_range=(1, 2), min_df=3, max_df=30,
                                       use_idf=False,
                                       lowercase=True)
        X = vectorizer.fit_transform(preprocessed_tweets)
        t0 = time()
        km.fit_transform(X)
        print("done in %0.3fs" % (time() - t0))
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        labels = km.predict(X)
        preprocessed_tweets_with_groups = defaultdict(list)
        original_tweets_with_groups = defaultdict(list)
        j = 0
        for i in labels:
            print(preprocessed_tweets[j])
            preprocessed_tweets_with_groups[str(i)].append(preprocessed_tweets[j])
            original_tweets_with_groups[str(i)].append(original_tweets[j])
            j = j + 1
        return preprocessed_tweets_with_groups, original_tweets_with_groups


