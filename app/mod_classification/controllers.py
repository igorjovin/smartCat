from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for, make_response, session
import StringIO
import json
from collections import defaultdict
from app.mod_classification.forms import PredictionsForm, AfterTrainingForm, CrossValidationForm
from app.mod_twitter.forms import IndexForm
import app.mod_classification.services as classification_service
from app.mod_classification.services import TweetClassifier
from app.mod_twitter.services import TweetPreprocessor
import app.mod_clustering.services as clustering_service

mod_classification = Blueprint('classification', __name__, url_prefix='/classification')
default_classifier_name = "Naive Bayes"

@mod_classification.route('/remove_tag', methods=['POST'])
def remove_tag_from_tweet():
    form = PredictionsForm(request.form)
    tweet_index = int(request.json['tweet_index'])
    tag = str(request.json['tag'])
    classes, tweets_with_predictions, indexes_with_tweets = classification_service.remove_tag_from_tweet(tweet_index, tag)
    return render_template("classification/predictions.html", form=form, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets, classes=classes)

@mod_classification.route('/add_tag', methods=['POST'])
def add_tag_to_tweet():
    form = PredictionsForm(request.form)
    desired_tag = str(request.json['desired_tag'])
    tweet_index = int(request.json['tweet_index'])
    classes, tweets_with_predictions, indexes_with_tweets = classification_service.add_tag_to_tweet(tweet_index, desired_tag)
    return render_template("classification/predictions.html", form=form, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets, classes=classes)

@mod_classification.route('/move_to_class_user', methods=['POST'])
def move_tweet_to_class_user():
    form = PredictionsForm(request.form)
    key = str(request.json['key'])
    hashtag = str(request.json['hashtag'])
    desired_key = str(request.json['desired_key'])
    tweet_index = int(request.json['index']) - 1
    tweets_with_predictions = classification_service.move_tweet_to_class(key, desired_key, tweet_index)
    classification_service.retrain_classifier(hashtag, default_classifier_name)
    return render_template("classification/predictions_user.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/new_group', methods=['POST'])
def create_new_group():
    form = PredictionsForm(request.form)
    desired_name= str(request.json['desired_name'])
    classification_service.create_new_class(desired_name) 
    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/choose_classifier', methods=['POST'])
def choose_classifier():
    classifier_name = str(request.json['classifier_name'])
    classification_service.select_classifier(classifier_name)
    return json.dumps("{Status: 'OK'}");
    
@mod_classification.route('/train')
def train():
    classification_service.train_classifier(clustering_service.preprocessed_tweets_with_groups, clustering_service.group_names, clustering_service.used_hashtag )
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
    if classifier_name == "":
        classifier_name = default_classifier_name
    classification_service.retrain_classifier(hashtag, classifier_name)
    form = AfterTrainingForm(request.form)
    return render_template("classification/after_training.html", form=form)

@mod_classification.route('/predict/', methods=['GET', 'POST'])
def predict():
    form = AfterTrainingForm(request.form)
    hashtag = str(request.json['hashtag'])
    classifier_name = str(request.json['classifier_name'])
    classes, tweets_with_predictions, indexes_with_tweets = classification_service.predict(hashtag, classifier_name, 100, 0, None)
    return render_template("classification/predictions.html", form=form, classes=classes, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets)

@mod_classification.route('/predict-user/', methods=['GET', 'POST'])
def predict_user():
    form = IndexForm(request.form)
    hashtag = str(request.json['hashtag'])
    filter_classes = list(request.json['filter_classes'])
    previous_filter_classes = list()
    classes_from_session = list()
    tweets_with_predictions_from_session = dict()
    indexes_with_tweets_from_session = dict()
    offset = 0
    if session.get('offset'):
        offset = session['offset']
    if session.get('filter_classes'):
        previous_filter_classes = session['filter_classes']
    if session.get('classes'):
        classes_from_session = session['classes']
    if session.get('tweets_with_predictions'):
        tweets_with_predictions_from_session = session['tweets_with_predictions']
    if session.get('indexes_with_tweets'):
        indexes_with_tweets_from_session = session['indexes_with_tweets']
    classes_from_session, tweets_with_predictions, indexes_with_tweets = classification_service.predict(hashtag, "Naive Bayes", 20, offset, filter_classes)
    if previous_filter_classes == filter_classes:
        classes_from_session.append(list(classes_from_session))
        tweets_with_predictions_from_session.update(tweets_with_predictions)
        indexes_with_tweets_from_session.update(indexes_with_tweets)
        print("UPDATED DICT %d" % len(indexes_with_tweets_from_session.keys()))
    else:
        classes_from_session = list(classes_from_session)
        tweets_with_predictions_from_session = tweets_with_predictions
        indexes_with_tweets_from_session = indexes_with_tweets
        offset = 0
    session['filter_classes'] = filter_classes
    print("LENGTH %d " % len(indexes_with_tweets.keys()))
    print("PREVIOUS OFFSET %d " % offset)
    offset = offset + len(indexes_with_tweets.keys())
    session['offset'] = offset
    session['classes'] = classes_from_session
    session['tweets_with_predictions'] = tweets_with_predictions
    session['indexes_with_tweets'] = indexes_with_tweets
    return render_template("classification/predictions_user.html", form=form, classes=classes_from_session, tweets_with_predictions=tweets_with_predictions_from_session, indexes_with_tweets=indexes_with_tweets_from_session)

@mod_classification.route('/prediction-classes/', methods=['GET', 'POST'])
def prediction_classes():
    form = IndexForm(request.form)
    hashtag = str(request.json['hashtag'])
    classes = classification_service.get_classes_from_dataset(hashtag)
    return render_template("classification/classes.html", form=form, classes=classes)

@mod_classification.route('/cross-validation', methods=['GET'])
def cross_validation():
    form = CrossValidationForm(request.form)
    return render_template("classification/cross_validation.html", form=form)

@mod_classification.route('/cross-validate', methods=['POST'])
def cross_validate():
    form = CrossValidationForm(request.form)
    hashtag = str(request.json['hashtag'])
    classifier_name = str(request.json['classifier_name'])
    accuracy, f1 = classification_service.cross_validate(hashtag, classifier_name)
    return render_template("classification/cross_validation_results.html", form=form, accuracy=accuracy, f1=f1)

@mod_classification.route('/switch_tweet_view', methods=['GET'])
def switch_tweet_view():
    form = PredictionsForm(request.form)
    tweets_with_predictions = classification_service.switch_tweet_view()
    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/save_to_dataset', methods=['POST'])
def save_to_dataset():
    form = AfterTrainingForm(request.form)
    hashtag = str(request.json['hashtag'])
    classifier_name = str(request.json['classifier_name'])
    if classifier_name == "":
        classifier_name = default_classifier_name
    classification_service.save_to_dataset(hashtag, classifier_name)
    return render_template("classification/after_training.html", form=form)

@mod_classification.route('/logout', methods=['GET'])
def invalidate_session():
    form = IndexForm(request.form)
    session.clear()
    return render_template("twitter/index_user.html", form=form)



