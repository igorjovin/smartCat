# Statement for enabling the development environment
DEBUG = True

# Define the application directory
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))#os.path.abspath(os.path.dirname(__file__))  

# Define the database - we are working with
# SQLite for this example
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
DATABASE_CONNECT_OPTIONS = {}

# Application threads. A common general assumption is
# using 2 per available processor cores - to handle
# incoming requests using one and performing background
# operations using the other.
THREADS_PER_PAGE = 2

# Enable protection agains *Cross-site Request Forgery (CSRF)*
CSRF_ENABLED     = True

# Use a secure, unique and absolutely secret key for
# signing the data. 
CSRF_SESSION_KEY = "secret4"

# Secret key for signing cookies
SECRET_KEY = "secret4"

#Twitter auth credentials
TWITTER_CONSUMER_KEY = '9anEflCDSYuUY36Wl2kecwkxe'
TWITTER_CONSUMER_SECRET = 'bsmj1WqCL8WgyGDfafLkzPOSg9p5QUWDyDgboUWLGeFWVWnsQY'
TWITTER_ACCESS_TOKEN = '56657567-xKPRh2Y1daGr7848aZjkGyQcE1OWjFKhkeEUxs2En'
TWITTER_ACCESS_SECRET = 'mve1oAOuGONdfpfZY2SgwGH4NzLoV89bk43v13Zsnhrfz'

APP_STATIC = os.path.join(BASE_DIR, 'app/static')

