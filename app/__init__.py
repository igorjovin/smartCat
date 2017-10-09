from flask import Flask, render_template

# Define the WSGI application object
app = Flask(__name__)

# Configurations
app.config.from_object('config')

# Import a module / component using its blueprint handler variable
from app.mod_twitter.controllers import mod_twitter as twitter_module

# Register blueprint(s)
app.register_blueprint(twitter_module)
# app.register_blueprint(xyz_module)