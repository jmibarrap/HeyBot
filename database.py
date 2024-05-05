import os

class Database:
    """
    This class managaes a simple memory based database.
    
    It handles the following operations:
    - Add a tweet
    - Get all tweets
    - Get a single tweet
    - Delete a tweet

    The main data structure has the following information:
    - tweet_id - unique identifier
    - date
    - time
    - tweet_text
    - classification

    """
    

    def __init__(self):
        self.data = {}
    
    def add_tweet(self, tweet_id, date, time, tweet_text):
        self.data[tweet_id] = { 'date': date, 'time': time, 'tweet_text': tweet_text}

    def get_all_tweets(self):
        return self.data

    def get_tweet(self, tweet_id):
        return self.data.get(tweet_id)

    def delete_tweet(self, tweet_id):
        self.data.pop(tweet_id)