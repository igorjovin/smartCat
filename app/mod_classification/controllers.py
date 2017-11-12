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
import app.mod_clustering.controllers as clust

mod_classification = Blueprint('classification', __name__, url_prefix='/classification')
default_classifier_name = "Naive Bayes"

@mod_classification.route('/remove_from_class', methods=['POST'])
def remove_tweet_from_class():
    form = PredictionsForm(request.form)
    key = str(request.json['key'])
    tweet_index = int(request.json['index']) - 1
    tweets_with_predictions = classification_service.remove_tweet_from_class(key, tweet_index)
    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/move_to_class', methods=['POST'])
def move_tweet_to_class():
    form = PredictionsForm(request.form)
    key = str(request.json['key'])
    desired_key = str(request.json['desired_key'])
    tweet_index = int(request.json['index']) - 1
    tweets_with_predictions = classification_service.move_tweet_to_class(key, desired_key, tweet_index)
    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

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
    classification_service.train_classifier(clust.preprocessed_tweets_with_groups, clust.group_names, clust.hashtag)
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
    tweets_with_predictions = classification_service.predict(hashtag, classifier_name)
    return render_template("classification/predictions.html", form=form, tweets=tweets_with_predictions)

@mod_classification.route('/predict-user/', methods=['GET', 'POST'])
def predict_user():
    form = IndexForm(request.form)
    hashtag = str(request.json['hashtag'])
    tweets_with_predictions = classification_service.predict(hashtag, default_classifier_name)
    return render_template("classification/predictions_user.html", form=form, tweets=tweets_with_predictions)

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


