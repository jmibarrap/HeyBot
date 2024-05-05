# from Model import Model
# from Preprocessing import Preprocessing
from selenium_scraper.scraper.twitter_scraper import Twitter_Scraper
import os
import sys
import random
import tweepy
import time

from dotenv import load_dotenv
load_dotenv()

class Bot:

    TWEET_CONTENTS = {1: "I'm sorry to hear that. Please contact our customer service at 1-800-123-4567", 2: "I'm glad you're happy with our service. We're always here to help you."}

    def __init__(self, model_weight_path="", database_data={}, sleep_time=300):
        #self.model = Model(model_weight_path)
        #self.preprocessing = Preprocessing()
        self.database_data = database_data
        self.tweets_to_read = {}
        self.sleep_time = sleep_time
        self.tweepy_client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        )
    
    def _call_scraper(self, **kwargs):
        USER_MAIL = kwargs["mail"] if "mail" in kwargs else os.getenv("TWITTER_MAIL")
        USER_UNAME = kwargs["user"] if "user" in kwargs else os.getenv("TWITTER_USERNAME")
        USER_PASSWORD = kwargs["password"] if "password" in kwargs else os.getenv("TWITTER_PASSWORD")

        if USER_UNAME is None:
            raise Exception("Twitter Username is required.")

        if USER_PASSWORD is None:
            raise Exception("Twitter Password is required.")

        tweet_type_args = []

        if "username" in kwargs :
            tweet_type_args.append(kwargs["username"])
        if "hashtag" in kwargs:
            tweet_type_args.append(kwargs["hashtag"])
        if "query" in kwargs:
            tweet_type_args.append(kwargs["query"])

        if len(tweet_type_args) > 1:
            print("Please specify only one of username, hashtag, or query.")
            sys.exit(1)

        if "latest" in kwargs and "top" in kwargs:
            print("Please specify either latest or top. Not both.")
            sys.exit(1)

        if USER_UNAME is not None and USER_PASSWORD is not None:
            scraper = Twitter_Scraper(
                mail=USER_MAIL,
                username=USER_UNAME,
                password=USER_PASSWORD,
            )
            scraper.login()
            scraper.scrape_tweets(
                max_tweets=kwargs["tweets"] if ("tweets" in kwargs) else 50,
                no_tweets_limit=kwargs["no_tweets_limit"] if ("no_tweets_limit" in kwargs) else False,
                scrape_username=kwargs["username"] if ("username" in kwargs) else None,
                scrape_hashtag=kwargs["hashtag"] if ("hashtag" in kwargs) else None,
                scrape_query=kwargs["query"] if ("query" in kwargs) else None,
                scrape_latest=kwargs["latest"] if ("latest" in kwargs) else False,
                scrape_top=kwargs["top"] if ("top" in kwargs) else True,
            )
            scraper.save_to_csv()
            if not scraper.interrupted:
                scraper.driver.close()
        else:
            print(
                "Missing Twitter username or password environment variables. Please check your .env file."
            )
            sys.exit(1)

    def _read_tweets_from_csv(self):
        tweets_folder_path = os.path.join(os.getcwd(), "tweets")
        for file in os.listdir(tweets_folder_path):
            if file.endswith(".csv"):
                with open(os.path.join(tweets_folder_path, file), "r") as f:
                    file_data = f.readlines()
                    for index, line in enumerate(file_data):
                        if index == 0:
                            continue
                        column_data = line.split(",")
                        tweet_id = column_data[-1]
                        if tweet_id not in self.database_data:
                            timestamp = column_data[2].split("t")
                            date = timestamp[0]
                            time = timestamp[1]
                            tweet_text = column_data[4]
                            self.tweets_to_read[tweet_id] = {'date': date, 'time': time, 'tweet_text': tweet_text}
                os.remove(os.path.join(tweets_folder_path, file))
        return self.database_data

    def _dummy_evaluate(self, tweet):
        return random.choice([0, 1, 2])

    def _get_actual_tweets(self):
        # self._call_scraper(tweets=10, query="heybanco (from:zAngelGa35156959)")
        self._call_scraper(tweets=10, username="elonmusk")
        self._read_tweets_from_csv()
        print(self.tweets_to_read) # Debugging purposes

    def _post_tweet_response(self, evaluation, tweet_id):
        response = self.TWEET_CONTENTS[evaluation]
        self.tweepy_client.create_tweet(text=response, in_reply_to_tweet_id=tweet_id)

    def _evaluate_tweets(self):
        for tweet_id, tweet_data in self.tweets_to_read.items():
            tweet_text = tweet_data['tweet_text']
            tweet_evaluation = self._dummy_evaluate(tweet_text)
            if tweet_evaluation != 0:
                self._post_tweet_response(tweet_evaluation, tweet_id)
            self.database_data[tweet_id] = { 'date': tweet_data['date'], 'time': tweet_data['time'], 'tweet_text': tweet_text, 'evaluation': tweet_evaluation}
        self.tweets_to_read = {}
        return self.database_data

    def run_bot(self):
        self._get_actual_tweets()
        self._evaluate_tweets()
        time.sleep(self.sleep_time)

if __name__ == "__main__":
    bot = Bot()
    bot.run_bot()