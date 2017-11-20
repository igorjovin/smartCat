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
import app.mod_twitter.services as twitter_service

import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
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

auth = OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)#, proxy="http://proxy.uns.ac.rs:8080")

preprocessed_tweets_with_groups = defaultdict(list)
original_tweets_with_groups = defaultdict(list)
tweets_to_show = defaultdict(list) #tweets which will be shown on the view - either original, or preprocessed
group_names = {}
group_list = list()
num_of_groups = 0
showing_original_tweets = True
used_hashtag = ""

def cluster(hashtag, num_of_clusters):
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global tweets_to_show
    global group_names
    global num_of_groups
    global used_hashtag

    group_names = {}
    num_of_groups = num_of_clusters
    hashtag = twitter_service.process_hashtag(hashtag)
    preprocessed_tweets = list()
    original_tweets = list()
    for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en").items(200):
        tweet_text = twitter_service.process(tweet, hashtag)
        if tweet_text not in preprocessed_tweets and len(tweet_text) > 1:
           preprocessed_tweets.append(tweet_text)
           original_tweets.append(tweet.text)

    kmeans = TwitterKMeans(num_of_clusters)
    preprocessed_tweets_with_groups, original_tweets_with_groups = kmeans.perform_clustering(preprocessed_tweets, original_tweets, num_of_clusters)
    tweets_to_show = original_tweets_with_groups
    used_hashtag = hashtag
    return group_names, tweets_to_show

def move_tweet_to_cluster(key, desired_key, tweet_index, is_copy):
    if tweet_index == -1:
        tweet_index = 0
    global group_names
    global num_of_groups
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global tweets_to_show

    if not all(str.isdigit(c) for c in desired_key):
        desired_name = desired_key.lower().strip().replace(" ", "_") 
        desired_key = int(num_of_groups)
        num_of_groups = num_of_groups + 1
        if desired_name not in group_names.values():
            group_names[str(desired_key)] = str(desired_name)

    tweet = preprocessed_tweets_with_groups[key][tweet_index]
    if not is_copy:
        del preprocessed_tweets_with_groups[key][tweet_index]
    preprocessed_tweets_with_groups[str(desired_key)].append(tweet)
    tweet = original_tweets_with_groups[key][tweet_index]
    if not is_copy:
        del original_tweets_with_groups[key][tweet_index]
    original_tweets_with_groups[str(desired_key)].append(tweet)

    tweets_to_show = preprocessed_tweets_with_groups
    if showing_original_tweets:
        tweets_to_show = original_tweets_with_groups
    return group_names, tweets_to_show

def remove_tweet_from_cluster(key, tweet_index):
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global group_names
    tweet = preprocessed_tweets_with_groups[key][tweet_index]
    del preprocessed_tweets_with_groups[key][tweet_index]
    tweet = original_tweets_with_groups[key][tweet_index]
    del original_tweets_with_groups[key][tweet_index]
    return group_names, tweets_to_show

def delete_cluster(key):
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global group_names
    del preprocessed_tweets_with_groups[key]
    del original_tweets_with_groups[key]
    if key in group_names.keys():
        del group_names[key]
    return group_names, tweets_to_show

def merge_cluster(key, desired_key):
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global group_names
    tweets = preprocessed_tweets_with_groups[key]
    del preprocessed_tweets_with_groups[key]
    preprocessed_tweets_with_groups[desired_key].extend(tweets)
    tweets = original_tweets_with_groups[key]
    del original_tweets_with_groups[key]
    original_tweets_with_groups[desired_key].extend(tweets)
    return group_names, tweets_to_show

def change_cluster_name(key, desired_name):
    global group_names
    global tweets_to_show
    if desired_name not in group_names.values():
        group_names[key] = desired_name.lower().strip().replace(" ", "_") 
    return group_names, tweets_to_show

def switch_tweet_view():
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global group_names
    global showing_original_tweets
    global tweets_to_show
    if showing_original_tweets:
        showing_original_tweets = False
        tweets_to_show = preprocessed_tweets_with_groups
    else:
        showing_original_tweets = True
        tweets_to_show = original_tweets_with_groups
    return group_names, tweets_to_show
