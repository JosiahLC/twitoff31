from os import name
from flask_sqlalchemy import SQLAlchemy
import spacy
import en_core_web_sm
import numpy as np
from sklearn.linear_model import LogisticRegression

db = SQLAlchemy()

# Creates a 'user' table
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True) # id as primary key
    name = db.Column(db.String(50), nullable=False) # user name
    tweet = db.Column(db.PickleType, nullable=True)
    tweet_vect = db.Column(db.PickleType, nullable=True)

    def __init__(self, name, tweet, tweet_vect):
        self.name = name
        self.tweet = tweet
        self.tweet_vect = tweet_vect
        
class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True) # id as primary key
     # user name
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    user = db.relationship("User", backref=db.backref('tweets', lazy=True))
    

    def __repr__(self):
        return "<Tweet: {}>".format(self.tweet)

def vectorize_tweet(twitext):
    """makes text a vector. retuens the vector"""
    nlp = en_core_web_sm.load()
    return nlp(twitext).vector
def sing_vects(twit):
    """makes an array of a series of texts. returns the array"""
    tweets_vec = np.array([vectorize_tweet(tweet.text)for tweet in twit])
    return tweets_vec
def sing_vect(twit):
    """makes an array of a series of texts. returns the array"""
    tweets_vec = np.array([vectorize_tweet(tweet)for tweet in twit])
    return tweets_vec
def comp_vects(zeros, ones):
    """combines two arrays of vectors, labels and fits a logistic regression to them. returns the regression"""
    vects = np.vstack([sing_vect(zeros), sing_vect(ones)])
    labels = np.concatenate([np.zeros(len(zeros)), np.ones( len(ones))])
    log_reg = LogisticRegression().fit(vects, labels)
    return log_reg
