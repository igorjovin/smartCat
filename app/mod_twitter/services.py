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
from collections import defaultdict
import pandas as pd
import nltk
import pickle
nltk.download('wordnet')
from nltk import WordNetLemmatizer

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
    hashtag = ""
    acronym_map = dict()
    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)

    def __init__(self, tweet, hashtag):
        self.tweet = tweet
        self.hashtag = hashtag
        reload(sys)
        sys.setdefaultencoding('utf-8')
        self.init_acronym_map()

    def perform_preprocessing(self):
        #ls = LancasterStemmer()
        lemmatizer = WordNetLemmatizer()
        self.emoji_pattern.sub(r'', self.tweet)
        tweet_parts = self.tweet.split(" ")
        preprocessed_tweet = ""
        for part in tweet_parts:
            #part = ls.stem(part)
            part = part.decode('utf-8')
            #part = self.manipulate_hashtag_words(part)
            part = part.lower().strip()
            part = self.remove_username(part)
            part = self.remove_url(part)
            part = self.replace_acronyms(part)
            part = part.lower()
            part = self.remove_stop_words(part)
            part = self.remove_hashtag(part)
            part = self.remove_punctuation(part)
            part = self.remove_numbers(part)
            part = lemmatizer.lemmatize(part)
            part = self.remove_stop_words(part)
            preprocessed_tweet += part + " "
        return preprocessed_tweet.strip()

    def remove_username(self, part):
        if part.startswith('@'):
            part = ""
        return part

    def remove_url(self, part):
        part = re.sub(r'^https?:\/\/.*[\r\n]*', "", part, flags=re.MULTILINE)
        return part

    def remove_punctuation(self, part):
        part = re.sub("[^\\w\\s]", "", part)
        return part

    def remove_numbers(self, part):
        part = re.sub('[0-9]+', "", part)
        return part

    def remove_stop_words(self, part):
        with open(os.path.join(APP_STATIC, 'stopwords.txt')) as f:
            lines = f.readlines()
        #part = part.strip()
        for line in lines:
            line = line.strip()
            if part == line:
                part = ""
                break
        return part

    def remove_hashtag(self, part):
        stripped_hashtag = self.hashtag.replace("#", "").strip()
        if part == self.hashtag or stripped_hashtag in part:
            part = ""
        return part

    def manipulate_hashtag_words(self, part):
        if '#' in part:
            part = part + " " + part
        return part

    def replace_acronyms(self, part):
        if part in self.acronym_map.keys():
            part = self.acronym_map[part]
        return part

    def init_acronym_map(self):
        with open(os.path.join(APP_STATIC, "acronyms.txt"),'r') as f:
             for line in f:
                line_parts = re.split(r'\t+', line)
                self.acronym_map[line_parts[0]] = line_parts[1]
