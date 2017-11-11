# Run a test server.
from app import app
import redis
r = redis.StrictRedis(host='localhost', port=6379, db=0)
app.run(host='0.0.0.0', port=8080, debug=True)