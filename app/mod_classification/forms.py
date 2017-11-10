# Import Form 
from flask.ext.wtf import Form

# Import Form elements
from wtforms import TextField, FieldList 

# Import Form validators
from wtforms.validators import Required, Email, EqualTo

class AfterTrainingForm(Form):

    hashtag  = TextField('hashtag', [Required(message='You must enter a hashtag')])

class PredictionsForm(Form):

    hashtag  = TextField('hashtag', [Required(message='You must enter a hashtag')])