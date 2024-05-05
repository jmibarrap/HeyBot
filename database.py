"""
A module containing a class representing a simple memory based database.

Authors:
- Jose Angel Garcia Gomez
- Pablo Gonzalez de la Parra
- Jose Maria Ibarra Perez
- Ana Martinez Barbosa

"""

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
        """
        Add a tweet to the database.

        Parameters:
        - tweet_id (str): ID of the tweet.
        - date (str): Date of the tweet.
        - time (str): Time of the tweet.
        - tweet_text (str): Text of the tweet.
        """
        self.data[tweet_id] = { 'date': date, 'time': time, 'tweet_text': tweet_text}

    def get_all_tweets(self):
        """
        Retrieve all tweets from the database.

        Returns:
        - data (dict): Dictionary containing all tweets.
        """
        return self.data

    def get_tweet(self, tweet_id):
        """
        Retrieve a specific tweet from the database.

        Parameters:
        - tweet_id (str): ID of the tweet.

        Returns:
        - tweet_data (dict): Data of the specified tweet.
        """
        return self.data.get(tweet_id)

    def delete_tweet(self, tweet_id):
        """
        Delete a tweet from the database.

        Parameters:
        - tweet_id (str): ID of the tweet.
        """
        self.data.pop(tweet_id)