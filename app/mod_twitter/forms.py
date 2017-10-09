# Import Form 
from flask.ext.wtf import Form

# Import Form elements
from wtforms import TextField 

# Import Form validators
from wtforms.validators import Required, Email, EqualTo

class IndexForm(Form):
    hashtag  = TextField('hashtag', [Required(message='You must enter a hashtag')])