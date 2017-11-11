# Import flask dependencies
from flask import Blueprint, request, render_template, \
                  flash, g, redirect, url_for, make_response
from app.mod_twitter.forms import IndexForm

# Define the blueprint: 'auth', set its url prefix: app.url/auth
mod_twitter = Blueprint('twitter', __name__, url_prefix='/twitter')

@mod_twitter.route('/index/', methods=['GET'])
def index():
    return render_template("twitter/index.html")

@mod_twitter.route('/index-admin/', methods=['GET'])
def index_admin():
    form = IndexForm(request.form)
    return render_template("twitter/index_admin.html", form=form)
