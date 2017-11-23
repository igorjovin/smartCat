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
previous_hashtag = ""
items_per_page = 10
items_to_return = 100
is_user = False

@mod_classification.route('/remove_tag', methods=['POST'])
def remove_tag_from_tweet():
    global is_user
    form = PredictionsForm(request.form)
    tweet_index = int(request.json['tweet_index'])
    tag = str(request.json['tag'])
    classes, tweets_with_predictions, indexes_with_tweets = classification_service.remove_tag_from_tweet(tweet_index, tag)
    if not is_user:
        return render_template("classification/predictions.html", form=form, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets, classes=classes)
    else:
        return render_template("classification/predictions_user.html", form=form, classes=classes, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets)

@mod_classification.route('/add_tag', methods=['POST'])
def add_tag_to_tweet():
    global is_user
    form = PredictionsForm(request.form)
    desired_tag = str(request.json['desired_tag'])
    tweet_index = int(request.json['tweet_index'])
    classes, tweets_with_predictions, indexes_with_tweets = classification_service.add_tag_to_tweet(tweet_index, desired_tag)
    if not is_user:
        return render_template("classification/predictions.html", form=form, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets, classes=classes)
    else:
        return render_template("classification/predictions_user.html", form=form, classes=classes, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets)

@mod_classification.route('/new_group', methods=['POST'])
def create_new_group():
    global is_user
    form = PredictionsForm(request.form)
    desired_name= str(request.json['desired_name'])
    classification_service.create_new_class(desired_name) 
    if not is_user:
        return render_template("classification/predictions.html", classes=classes, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets)
    else:
        return render_template("classification/predictions_user.html", form=form, classes=classes, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets)

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
    global is_user
    form = AfterTrainingForm(request.form)
    hashtag = str(request.json['hashtag'])
    classifier_name = str(request.json['classifier_name'])
    is_user = False
    classes, tweets_with_predictions, indexes_with_tweets = classification_service.predict(hashtag, classifier_name, items_to_return, None)
    return render_template("classification/predictions.html", form=form, classes=classes, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets)

@mod_classification.route('/predict-user/', methods=['GET', 'POST'])
def predict_user():
    global is_user
    global previous_hashtag
    if previous_hashtag != "":
        classification_service.save_to_dataset(previous_hashtag, default_classifier_name)
    form = IndexForm(request.form)
    hashtag = str(request.json['hashtag'])
    filter_classes = list(request.json['filter_classes'])
    previous_filter_classes = list()
    classes_from_session = list()
    tweets_with_predictions_from_session = dict()
    indexes_with_tweets_from_session = dict()
    offset = items_per_page
    if session.get('offset'):
        offset = session['offset']
    if session.get('filter_classes'):
        previous_filter_classes = session['filter_classes']
    if session.get('classes'):
        classes_from_session = session['classes']
    classes_from_session, tweets_with_predictions, indexes_with_tweets = classification_service.predict(hashtag, default_classifier_name, 100, filter_classes)
    if previous_filter_classes != filter_classes:
        offset = items_per_page
    session['filter_classes'] = filter_classes
    offset = offset + items_per_page
    session['offset'] = offset
    session['classes'] = classes_from_session
    previous_hashtag = hashtag
    is_user = True
    return render_template("classification/predictions_user.html", form=form, classes=classes_from_session, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets)

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
    classes, tweets_with_predictions, indexes_with_tweets = classification_service.switch_tweet_view()
    return render_template("classification/predictions.html", form=form, classes=classes, tweets_with_predictions=tweets_with_predictions, indexes_with_tweets=indexes_with_tweets)

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
    global previous_hashtag
    form = IndexForm(request.form)
    if previous_hashtag != "":
        classification_service.save_to_dataset(previous_hashtag, default_classifier_name)
    session.clear()
    return render_template("twitter/index_user.html", form=form)



