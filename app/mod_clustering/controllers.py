# Import flask dependencies
from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for, make_response, session
import StringIO
import csv
import tweepy
import json
import os
from tweepy import OAuthHandler
from tweepy import Stream
from collections import defaultdict
from config import BASE_DIR, APP_STATIC, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET

# Import module forms
from app.mod_clustering.forms import ClusterResultsForm
from app.mod_twitter.services import TweetPreprocessor
from app.mod_clustering.services import TwitterKMeans
 
auth = OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
auth.set_access_token(TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)#, proxy="http://proxy.uns.ac.rs:8080")

# Define the blueprint: 'auth', set its url prefix: app.url/auth
mod_clustering = Blueprint('clustering', __name__, url_prefix='/clustering')
preprocessed_tweets_with_groups = defaultdict(list)
original_tweets_with_groups = defaultdict(list)
tweets_to_show = defaultdict(list) #tweets which will be shown on the view - either original, or preprocessed
group_names = {}
group_list = list()
num_of_groups = 0
hashtag = ""
showing_original_tweets = True

def process_hashtag(hashtag):
    if "#" not in hashtag:
        hashtag = "#" + hashtag
    if " " in hashtag: #a user has entered white spaces
        hashtag = hashtag.replace(" ", "")
    return hashtag

def process(tweet, hashtag):
    preprocessed_tweet = TweetPreprocessor(tweet.text, hashtag).perform_preprocessing()
    return preprocessed_tweet

def update_session():
    global preprocessed_tweets_with_groups
    global hashtag
    global group_names
    print("HASHTAG " + hashtag)
    session['preprocessed_tweets_with_groups'] = preprocessed_tweets_with_groups
    session['hashtag'] = hashtag
    session['group_names'] = group_names
    session.modified = True

@mod_clustering.route('/move_to_cluster', methods=['POST'])
def move_tweet_to_cluster():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    desired_key = str(request.json['desired_key'])
    tweet_index = int(request.json['index']) - 1
    if tweet_index == -1:
        tweet_index = 0

    global group_names
    global num_of_groups
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global tweets_to_show

    if not all(str.isdigit(c) for c in desired_key):
        desired_name = desired_key.lower().strip()
        desired_key = int(num_of_groups)
        num_of_groups = num_of_groups + 1
        if desired_name not in group_names.values():
            group_names[str(desired_key)] = str(desired_name)

    tweet = preprocessed_tweets_with_groups[key][tweet_index]
    del preprocessed_tweets_with_groups[key][tweet_index]
    preprocessed_tweets_with_groups[str(desired_key)].append(tweet)

    tweet = original_tweets_with_groups[key][tweet_index]
    del original_tweets_with_groups[key][tweet_index]
    original_tweets_with_groups[str(desired_key)].append(tweet)

    tweets_to_show = preprocessed_tweets_with_groups
    if showing_original_tweets:
        tweets_to_show = original_tweets_with_groups

    update_session()

    return render_template("clustering/cluster_results.html", form=form, tweets=tweets_to_show, group_names=group_names)

@mod_clustering.route('/remove_from_cluster', methods=['POST'])
def remove_tweet_from_cluster():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    tweet_index = int(request.json['index']) - 1

    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global group_names
    tweet = preprocessed_tweets_with_groups[key][tweet_index]
    del preprocessed_tweets_with_groups[key][tweet_index]

    tweet = original_tweets_with_groups[key][tweet_index]
    del original_tweets_with_groups[key][tweet_index]

    update_session()

    return render_template("clustering/cluster_results.html", form=form, tweets=tweets_to_show, group_names=group_names)

@mod_clustering.route('/delete_cluster', methods=['DELETE'])
def delete_cluster():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global group_names
    del preprocessed_tweets_with_groups[key]
    del original_tweets_with_groups[key]
    del group_names[key]

    update_session()

    return render_template("clustering/cluster_results.html", form=form, tweets=tweets_to_show, group_names=group_names)

@mod_clustering.route('/merge_cluster', methods=['POST'])
def merge_cluster():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    desired_key = str(request.json['desired_key'])

    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global group_names
    tweets = preprocessed_tweets_with_groups[key]
    del preprocessed_tweets_with_groups[key]
    preprocessed_tweets_with_groups[desired_key].extend(tweets)

    tweets = original_tweets_with_groups[key]
    del original_tweets_with_groups[key]
    original_tweets_with_groups[desired_key].extend(tweets)

    update_session()

    return render_template("clustering/cluster_results.html", form=form, tweets=tweets_to_show, group_names=group_names)

@mod_clustering.route('/switch_tweet_view', methods=['GET'])
def switch_tweet_view():
    form = ClusterResultsForm(request.form)
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

    return render_template("clustering/cluster_results.html", form=form, tweets=tweets_to_show, group_names=group_names)

@mod_clustering.route('/change_group_name', methods=['POST'])
def change_group_name():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    desired_name= str(request.json['desired_name'])

    global group_names
    global hashtag
    global preprocessed_tweets_with_groups
    if desired_name not in group_names.values():
        group_names[key] = desired_name.lower().strip().replace(" ", "_") 
        session['preprocessed_tweets_with_groups'] = preprocessed_tweets_with_groups
        session['hashtag'] = hashtag
        session['group_names'] = group_names
        g.hashtag = hashtag
        g.group_names = group_names
        g.preprocessed_tweets_with_groups = preprocessed_tweets_with_groups
        #update_session()

    return render_template("clustering/cluster_results.html", form=form, tweets=tweets_to_show, group_names=group_names)

# Set the route and accepted methods
@mod_clustering.route('/search/', methods=['GET', 'POST'])
def cluster_results():
    form = ClusterResultsForm(request.form)
    global preprocessed_tweets_with_groups
    global original_tweets_with_groups
    global tweets_to_show
    global group_names
    global hashtag
    global num_of_groups
    group_names = {}
    hashtag = str(request.json['hashtag'])
    num_of_clusters = int(request.json['num_of_clusters'])
    num_of_groups = num_of_clusters
    hashtag = process_hashtag(hashtag)
    preprocessed_tweets = list()
    original_tweets = list()
    for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en").items(200):
        tweet_text = process(tweet, hashtag)
        if tweet_text not in preprocessed_tweets and len(tweet_text) > 1:
    	   preprocessed_tweets.append(tweet_text)
           original_tweets.append(tweet.text)

    kmeans = TwitterKMeans(num_of_clusters)
    preprocessed_tweets_with_groups, original_tweets_with_groups = kmeans.perform_clustering(preprocessed_tweets, original_tweets, num_of_clusters)
    tweets_to_show = original_tweets_with_groups

    #update_session()
    session['preprocessed_tweets_with_groups'] = preprocessed_tweets_with_groups
    session['hashtag'] = hashtag
    session['group_names'] = group_names
    print("HASHTAG IN SESSION " + session['hashtag'])

    return render_template("clustering/cluster_results.html", form=form, tweets=tweets_to_show, group_names = group_names)

