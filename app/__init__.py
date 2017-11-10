from flask import Flask, render_template

# Define the WSGI application object
app = Flask(__name__)

# Configurations
app.config.from_object('config')

# Import a module / component using its blueprint handler variable
from app.mod_twitter.controllers import mod_twitter as twitter_module
from app.mod_classification.controllers import mod_classification as classification_module
from app.mod_clustering.controllers import mod_clustering as clustering_module

# Register blueprint(s)
app.register_blueprint(twitter_module)
app.register_blueprint(classification_module)
app.register_blueprint(clustering_module)