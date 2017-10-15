# Import flask dependencies
from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream

# Import module forms
from app.mod_twitter.forms import IndexForm
from app.mod_twitter.services import TwitterKMeans, TweetPreprocessor

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
tweets = list() #dataset

def process(tweet):
    preprocessed_tweet = TweetPreprocessor(tweet.text).perform_preprocessing()
    write_to_file(tweet)
    #print preprocessed_tweet
    return preprocessed_tweet
    #if preprocessed_tweet not in tweets:

def write_to_file(tweet):
    try:
        with open('python.json', 'a') as f:
            f.write(tweet.text)
    except Exception as e:
        print("Error on_data: %s" % str(e))

# Set the route and accepted methods
@mod_twitter.route('/search/', methods=['GET', 'POST'])
def index():

    # If form is submitted
    form = IndexForm(request.form)
    tweets = list()
    #form.hashtag.data
    for tweet in tweepy.Cursor(api.search, q="cassandra", lang="en").items(200):
        tweet_text = process(tweet)
        if tweet_text not in tweets:
    	   tweets.append(tweet_text)

    kmeans = TwitterKMeans(5)
    tweets_with_groups = kmeans.perform_clustering(tweets, 5)
    #for tweet in tweepy.Cursor(api.user_timeline).items():	

    return render_template("twitter/index.html", form=form, tweets = tweets_with_groups)