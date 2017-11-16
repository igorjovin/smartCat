# Import flask dependencies
from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for, make_response, session
import StringIO
import json
# Import module forms
from app.mod_clustering.forms import ClusterResultsForm
import app.mod_clustering.services as clustering_service

# Define the blueprint: 'auth', set its url prefix: app.url/auth
mod_clustering = Blueprint('clustering', __name__, url_prefix='/clustering')

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
    is_copy_str = str(request.json['is_copy'])
    print("IS COPY " + is_copy_str)
    is_copy = True
    if is_copy_str == "False":
        is_copy = False
    tweet_index = int(request.json['index']) - 1
    group_names, tweets = clustering_service.move_tweet_to_cluster(key, desired_key, tweet_index, is_copy)
    return render_template("clustering/cluster_results.html", form=form, tweets=tweets, group_names=group_names)

@mod_clustering.route('/remove_from_cluster', methods=['POST'])
def remove_tweet_from_cluster():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    tweet_index = int(request.json['index']) - 1
    group_names, tweets = clustering_service.remove_tweet_from_cluster(key, tweet_index)
    return render_template("clustering/cluster_results.html", form=form, tweets=tweets, group_names=group_names)

@mod_clustering.route('/delete_cluster', methods=['DELETE'])
def delete_cluster():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    group_names, tweets = clustering_service.delete_cluster(key)
    return render_template("clustering/cluster_results.html", form=form, tweets=tweets, group_names=group_names)

@mod_clustering.route('/merge_cluster', methods=['POST'])
def merge_cluster():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    desired_key = str(request.json['desired_key'])
    group_names, tweets = clustering_service.merge_cluster(key, desired_key)
    return render_template("clustering/cluster_results.html", form=form, tweets=tweets, group_names=group_names)

@mod_clustering.route('/switch_tweet_view', methods=['GET'])
def switch_tweet_view():
    form = ClusterResultsForm(request.form)
    group_names, tweets = clustering_service.switch_tweet_view()
    return render_template("clustering/cluster_results.html", form=form, tweets=tweets, group_names=group_names)

@mod_clustering.route('/change_cluster_name', methods=['POST'])
def change_cluster_name():
    form = ClusterResultsForm(request.form)
    key = str(request.json['key'])
    desired_name= str(request.json['desired_name'])
    group_names, tweets = clustering_service.change_cluster_name(key, desired_name)
    return render_template("clustering/cluster_results.html", form=form, tweets=tweets, group_names=group_names)

@mod_clustering.route('/search/', methods=['GET', 'POST'])
def cluster_results():
    form = ClusterResultsForm(request.form)
    hashtag = str(request.json['hashtag'])
    num_of_clusters = int(request.json['num_of_clusters'])
    group_names, tweets = clustering_service.cluster(hashtag, num_of_clusters)
    return render_template("clustering/cluster_results.html", form=form, tweets=tweets, group_names = group_names)

