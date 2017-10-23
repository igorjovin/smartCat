# Import flask dependencies
from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for, make_response
import StringIO
import csv
import tweepy
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
api = tweepy.API(auth)

# Define the blueprint: 'auth', set its url prefix: app.url/auth
mod_twitter = Blueprint('twitter', __name__, url_prefix='/twitter')
tweets_with_groups = defaultdict(list)
classifier = TweetClassifier()

def process(tweet):
    preprocessed_tweet = TweetPreprocessor(tweet.text).perform_preprocessing()
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

@mod_twitter.route('/export')
def export():
    si = StringIO.StringIO()
    cw = csv.writer(si, delimiter=',')
    global tweets_with_groups
    for label, value in tweets_with_groups.iteritems():
        print("Label %s" % label)
        for item in value:
            #labeled_item = str(label) + ", " + str(item.encode('utf-8'))
            cw.writerow([str(label), str(item.encode('utf-8'))])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=dataset.csv"
    output.headers["Content-type"] = "text/csv"
    return output

@mod_twitter.route('/train')
def train():
    filepath = os.path.join(os.path.join(APP_STATIC, 'dataset.csv'))
    f = open(filepath, "a")
    cw = writer=csv.writer(f)
    global tweets_with_groups
    for label, value in tweets_with_groups.iteritems():
        print("Label %s" % label)
        for item in value:
            #labeled_item = str(label) + ", " + str(item.encode('utf-8'))
            cw.writerow([str(label), str(item.encode('utf-8'))])
    f.close()
    global classifier
    classifier.classify()
    form = AfterTrainingForm(request.form)
    return render_template("twitter/after_training.html", form=form)

@mod_twitter.route('/predict/', methods=['GET', 'POST'])
def predict():
    form = PredictionsForm(request.form)
    global classifier
    tweets = list()
    for tweet in tweepy.Cursor(api.search, q=form.hashtag.data, lang="en").items(100):
        tweet_text = process(tweet)
        if tweet_text not in tweets:
           tweets.append(tweet_text)

    classifier.predict(tweets)
    return render_template("twitter/predictions.html", form=form)

# Set the route and accepted methods
@mod_twitter.route('/search/', methods=['GET', 'POST'])
def cluster_results():
    form = ClusterResultsForm(request.form)
    global tweets_with_groups
    tweets = list()
    for tweet in tweepy.Cursor(api.search, q=form.hashtag.data, lang="en").items(200):
        tweet_text = process(tweet)
        if tweet_text not in tweets:
    	   tweets.append(tweet_text)

    kmeans = TwitterKMeans(int(form.num_of_clusters.data))
    tweets_with_groups = kmeans.perform_clustering(tweets, int(form.num_of_clusters.data))

    return render_template("twitter/cluster_results.html", form=form, tweets=tweets_with_groups)