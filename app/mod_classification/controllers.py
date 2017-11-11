from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for, make_response, session
import StringIO
import csv
import tweepy
import json
import os
import pandas as pd
from config import BASE_DIR, APP_STATIC, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
from tweepy import OAuthHandler
from tweepy import Stream
from collections import defaultdict
# Import module forms
from app.mod_classification.forms import PredictionsForm, AfterTrainingForm, CrossValidationForm
from app.mod_classification.services import TweetClassifier
from app.mod_twitter.services import TweetPreprocessor
import app.mod_clustering.controllers as clust
 
auth = OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)#, proxy="http://proxy.uns.ac.rs:8080")

mod_classification = Blueprint('classification', __name__, url_prefix='/classification')

tweets_with_predictions = defaultdict(list)
new_group_names = {}
group_list = list()
classifier = TweetClassifier("Naive Bayes")
chosen_hashtag = ""

def process_hashtag(hashtag):
    if "#" not in hashtag:
        hashtag = "#" + hashtag
    if " " in hashtag: #a user has entered white spaces
        hashtag = hashtag.replace(" ", "")
    return hashtag

def process(tweet, hashtag):
    preprocessed_tweet = TweetPreprocessor(tweet.text, hashtag).perform_preprocessing()
    return preprocessed_tweet

@mod_classification.route('/remove_from_class', methods=['POST'])
def remove_tweet_from_class():
    form = PredictionsForm(request.form)
    key = str(request.json['key'])
    tweet_index = int(request.json['index']) - 1

    global tweets_with_predictions
    global group_names
    tweet = tweets_with_predictions[key][tweet_index]
    del tweets_with_predictions[key][tweet_index]

    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/move_to_class', methods=['POST'])
def move_tweet_to_class():
    form = PredictionsForm(request.form)
    key = str(request.json['key'])
    desired_key = str(request.json['desired_key'])
    tweet_index = int(request.json['index']) - 1

    global tweets_with_predictions
    global group_list
    if desired_key not in group_list:
        group_list.append(desired_key)
    
    tweet = tweets_with_predictions[key][tweet_index]
    del tweets_with_predictions[key][tweet_index]
    tweets_with_predictions[desired_key].append(tweet)

    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/new_group', methods=['POST'])
def create_new_group():
    form = PredictionsForm(request.form)
    desired_name= str(request.json['desired_name'])

    global group_names
    group_names[len(group_names)] = desired_name 

    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/choose_classifier', methods=['POST'])
def choose_classifier():
    classifier_name = str(request.json['classifier_name'])
    global classifier
    classifier = TweetClassifier(classifier_name)
    return json.dumps("{Status: 'OK'}");
    
@mod_classification.route('/train')
def train():
    file_name = 'dataset-' + clust.hashtag + '.csv'
    filepath = os.path.join(os.path.join(APP_STATIC, file_name))
    f = open(filepath, "a")
    cw = csv.writer(f)
    global new_group_names
    new_group_names = clust.group_names

    for label, value in clust.preprocessed_tweets_with_groups.iteritems():
        for item in value:
            #labeled_item = str(label) + ", " + str(item.encode('utf-8'))
            cw.writerow([str(new_group_names[label]), str(item.encode('utf-8'))])
    f.close()
    global classifier
    classifier.classify(clust.hashtag)
    form = AfterTrainingForm(request.form)
    return render_template("classification/after_training.html", form=form)

@mod_classification.route('/after_training', methods=['GET'])
def after_training():
    form = AfterTrainingForm(request.form)
    return render_template("classification/after_training.html", form=form)

@mod_classification.route('/retrain/', methods=['POST'])
def retrain():
    hashtag = str(request.json['hashtag'])
    classifier_name = str(request.json['classifier_name'])
    file_name = 'dataset-' + hashtag + '.csv'
    filepath = os.path.join(os.path.join(APP_STATIC, file_name))
    tweets_to_append = get_tweets_to_append(hashtag)
    f = open(filepath, "a")
    cw = csv.writer(f)
    for label, value in tweets_to_append.iteritems():
        for item in value:
            cw.writerow([str(label), str(item.encode('utf-8'))])
    f.close()
    global classifier
    classifier.classify(hashtag)
    form = AfterTrainingForm(request.form)
    return render_template("classification/after_training.html", form=form)

@mod_classification.route('/predict/', methods=['GET', 'POST'])
def predict():
    form = AfterTrainingForm(request.form)
    global classifier
    global tweets_with_predictions
    global group_list
    hashtag = str(request.json['hashtag'])
    classifier_name = str(request.json['classifier_name'])
    hashtag = process_hashtag(hashtag)
    tweets = list()
    for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en").items(100):
        tweet_text = process(tweet, hashtag)
        if tweet_text not in tweets:
           tweets.append(tweet_text)

    classifier = TweetClassifier(classifier_name)
    tweets_with_predictions = classifier.predict(tweets, hashtag)
    if tweets_with_predictions is None:
        message = "There is no classifier " + classifier_name
        print(message)
    else:
        group_list = list(tweets_with_predictions.keys())

    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/cross-validation', methods=['GET'])
def cross_validation():
    form = CrossValidationForm(request.form)
    return render_template("classification/cross_validation.html", form=form)

@mod_classification.route('/cross-validate', methods=['POST'])
def cross_validate():
    form = CrossValidationForm(request.form)
    hashtag = str(request.json['hashtag'])
    classifier_name = str(request.json['classifier_name'])
    hashtag = process_hashtag(hashtag)
    classifier = TweetClassifier(classifier_name)
    accuracy, f1 = classifier.cross_validate(hashtag)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    accuracy = round(accuracy * 100.0, 2)
    f1 = round(f1 * 100.0, 2)
    return render_template("classification/cross_validation_results.html", form=form, accuracy=accuracy, f1=f1)

def get_tweets_to_append(hashtag):
    global tweets_with_predictions
    file_name = 'dataset-' + hashtag + '.csv'
    filepath = os.path.join(os.path.join(APP_STATIC, file_name))
    existing_tweets = pd.read_csv(filepath, header=None, names=["label", "tweet"])
    existing_tweet_list = existing_tweets["tweet"].tolist()
    new_tweets_with_predictions = defaultdict(list)
    for label, value in tweets_with_predictions.iteritems():
        for item in value:
            if item not in existing_tweet_list:
                print("TO APPEND " + item)
                new_tweets_with_predictions[label].append(item)

    return new_tweets_with_predictions



