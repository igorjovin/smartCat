# Import flask dependencies
from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for, make_response
import StringIO
import csv
import tweepy
import json
from tweepy import OAuthHandler
from tweepy import Stream
from collections import defaultdict
import os
from config import BASE_DIR, APP_STATIC

# Import module forms
from app.mod_twitter.forms import IndexForm, ClusterResultsForm, PredictionsForm, AfterTrainingForm
from app.mod_twitter.services import TwitterKMeans, TweetPreprocessor, TweetClassifier

#Twitter auth credentials
consumer_key = '9anEflCDSYuUY36Wl2kecwkxe'
consumer_secret = 'bsmj1WqCL8WgyGDfafLkzPOSg9p5QUWDyDgboUWLGeFWVWnsQY'
access_token = '56657567-xKPRh2Y1daGr7848aZjkGyQcE1OWjFKhkeEUxs2En'
access_secret = 'mve1oAOuGONdfpfZY2SgwGH4NzLoV89bk43v13Zsnhrfz'
 
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)#, proxy="http://proxy.uns.ac.rs:8080")

# Define the blueprint: 'auth', set its url prefix: app.url/auth
mod_twitter = Blueprint('twitter', __name__, url_prefix='/twitter')
tweets_with_groups = defaultdict(list)
group_names = {}
classifier = TweetClassifier()
hashtag = ""

def process_hashtag(hashtag):
    if "#" not in hashtag:
        hashtag = "#" + hashtag
    if " " in hashtag: #a user has entered white spaces
        hashtag = hashtag.replace(" ", "")
    return hashtag


def process(tweet, hashtag):
    preprocessed_tweet = TweetPreprocessor(tweet.text, hashtag).perform_preprocessing()
    write_to_file(tweet)
    #print preprocessed_tweet
    return preprocessed_tweet

def write_to_file(tweet):
    try:
        with open('python.json', 'a') as f:
            f.write(tweet.text)
    except Exception as e:
        print("Error on_data: %s" % str(e))

@mod_twitter.route('/index/', methods=['GET'])
def index():
    form = IndexForm(request.form)
    return render_template("twitter/index.html", form=form)

@mod_twitter.route('/move', methods=['POST'])
def move():
    form = ClusterResultsForm(request.form)
    key = int(request.json['key'])
    desired_key = int(request.json['desired_key'])
    tweet_index = int(request.json['index']) - 1

    global tweets_with_groups
    global group_names
    tweet = tweets_with_groups[key][tweet_index]
    del tweets_with_groups[key][tweet_index]
    tweets_with_groups[desired_key].append(tweet)

    return render_template("twitter/cluster_results.html", form=form, tweets=tweets_with_groups, group_names=group_names)

@mod_twitter.route('/change_group_name', methods=['POST'])
def change_group_name():
    form = ClusterResultsForm(request.form)
    key = int(request.json['key'])
    desired_name= str(request.json['desired_name'])

    global group_names
    group_names[key] = desired_name 

    return render_template("twitter/cluster_results.html", form=form, tweets=tweets_with_groups, group_names=group_names)

@mod_twitter.route('/export')
def export():
    si = StringIO.StringIO()
    cw = csv.writer(si, delimiter=',')
    global tweets_with_groups
    for label, value in tweets_with_groups.iteritems():
        for item in value:
            cw.writerow([str(label), str(item.encode('utf-8'))])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=dataset.csv"
    output.headers["Content-type"] = "text/csv"
    return output
    
@mod_twitter.route('/train')
def train():
    global hashtag
    file_name = 'dataset-' + hashtag + '.csv'
    filepath = os.path.join(os.path.join(APP_STATIC, file_name))
    f = open(filepath, "a")
    cw = csv.writer(f)
    global tweets_with_groups
    for label, value in tweets_with_groups.iteritems():
        print("Label %s" % label)
        for item in value:
            #labeled_item = str(label) + ", " + str(item.encode('utf-8'))
            cw.writerow([str(label), str(item.encode('utf-8'))])
    f.close()
    global classifier
    classifier.classify(hashtag)
    form = AfterTrainingForm(request.form)
    return render_template("twitter/after_training.html", form=form)

@mod_twitter.route('/predict/', methods=['GET', 'POST'])
def predict():
    form = PredictionsForm(request.form)
    global classifier
    hashtag = process_hashtag(form.hashtag.data)
    tweets = list()
    for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en").items(100):
        tweet_text = process(tweet, hashtag)
        if tweet_text not in tweets:
           tweets.append(tweet_text)

    tweets_with_predictions = classifier.predict(tweets)
    return render_template("twitter/predictions.html", form=form, tweets=tweets_with_predictions)

# Set the route and accepted methods
@mod_twitter.route('/search/', methods=['GET', 'POST'])
def cluster_results():
    form = ClusterResultsForm(request.form)
    global tweets_with_groups
    global group_names
    global hashtag
    group_names = {}
    hashtag = str(request.json['hashtag'])
    num_of_clusters = int(request.json['num_of_clusters'])
    hashtag = process_hashtag(hashtag)
    tweets = list()
    for tweet in tweepy.Cursor(api.search, q=hashtag, lang="en").items(200):
        tweet_text = process(tweet, hashtag)
        if tweet_text not in tweets:
    	   tweets.append(tweet_text)

    kmeans = TwitterKMeans(num_of_clusters)
    tweets_with_groups = kmeans.perform_clustering(tweets, num_of_clusters)

    return render_template("twitter/cluster_results.html", form=form, tweets=tweets_with_groups, group_names = group_names)