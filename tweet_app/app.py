import os
from flask import Flask, render_template, request
from .model import  db, User, vectorize_tweet, sing_vects, comp_vects
import tweepy

"""Create and configure an instance of the flask application"""
app = Flask(__name__)

# configure app
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv['DATABASE_URI']
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


# initialize database
db.init_app(app)

# create table(s)
with app.app_context():
    db.create_all()

TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_KEY_SECRET = os.getenv("TWITTER_API_KEY_SECRET")
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_KEY_SECRET)
twitter = tweepy.API(auth)


# ROOT ROUTE
@app.route('/', methods=["GET", "POST"])
def main():
    name = request.form.get("name")   
  
    """Base view"""
   
    if name:
        test = twitter.get_user(name)
        """gets twitter user"""
        take2 = test.screen_name
        test2 = test.timeline(count=200,
        exclude_replies=True, include_rts=False, tweet_mode="Extended")
        """gets twitter users tweets"""
        alluser = User(name=take2, tweet = test2, tweet_vect = sing_vects(test2))
        """adds the user, tweets, and tweet vectors to the datatbase"""
        db.session.add(alluser)
        db.session.commit()

    text = request.form.get("text")
    zeros = request.form.get("user_zeros")
    ones = request.form.get("user_ones")

    if zeros and ones and text:
        log = comp_vects(zeros, ones)
        result_ = vectorize_tweet(text)
        testing = result_.reshape(1, -1)
        test = log.predict(testing)
        if test == 1:
            tests= ones
        elif test == 0:
            tests = zeros
        return render_template("magic.html", results = tests)

    users = User.query.all()
    
    return render_template("home.html", users=users)

@app.route('/iris')
def iris():
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0, solver="lbfgs",
    multi_class='multinominal').fit(X, y)
    return str(clf.predict(X[:2, :]))

if __name__ == "__main__":
    app.run()
