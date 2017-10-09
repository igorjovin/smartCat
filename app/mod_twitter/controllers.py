# Import flask dependencies
from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for

# Import module forms
from app.mod_twitter.forms import IndexForm

# Define the blueprint: 'auth', set its url prefix: app.url/auth
mod_twitter = Blueprint('twitter', __name__, url_prefix='/twitter')

# Set the route and accepted methods
@mod_twitter.route('/search/', methods=['GET', 'POST'])
def index():

    # If form is submitted
    form = IndexForm(request.form)

    return render_template("twitter/index.html", form=form)