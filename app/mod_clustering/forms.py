# Import Form 
from flask.ext.wtf import Form

# Import Form elements
from wtforms import TextField, FieldList 

# Import Form validators
from wtforms.validators import Required, Email, EqualTo

class IndexForm(Form):

    hashtag  = TextField('hashtag', [Required(message='You must enter a hashtag')])
    num_of_clusters = TextField('num_of_clusters', [Required(message='You must enter number of clusters')])

class ClusterResultsForm(Form):

    hashtag  = TextField('hashtag', [Required(message='You must enter a hashtag')])
    num_of_clusters = TextField('num_of_clusters', [Required(message='You must enter number of clusters')])
