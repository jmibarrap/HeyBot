# from predictor import BertModelWrapper
from selenium_scraper.scraper.twitter_scraper import Twitter_Scraper
import os
import random
import tweepy
import time
import csv

from dotenv import load_dotenv
load_dotenv()

class Bot:

    TWEET_CONTENTS = {1: "I'm sorry to hear that. Please contact our customer service", 2: "I'm glad you're happy with our service. We're always here to help you."}

    def __init__(self, model_weight_path="", database_data={}, sleep_time=300):
        #self.model = Model(model_weight_path)
        self.database_data = database_data
        self.tweets_to_read = {}
        self.sleep_time = sleep_time
        self.tweepy_client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
        )
    
    def _read_csv_to_string(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            csv_data = '\n'.join(','.join(row) for row in reader)
        return csv_data

    def _clean_csv_data_string(self, csv_data):
        csv_data = csv_data.split("\n")
        csv_data = csv_data[1:]
        csv_data = list(filter(None, csv_data))
        for index, line in enumerate(csv_data):
            if line[-1].isdigit():
                twitter_id = line.split(",")[-1].split(":")[1]
                csv_data[index] =  ','.join(line.split(",")[:-1]+[twitter_id]) + "\n"
            else: 
                csv_data[index] = line + ","
        csv_data = ''.join(csv_data)
        csv_data = csv_data.split("\n")
        csv_data = [line.split(",") for line in csv_data]
        csv_data = [row for row in csv_data if len(row) > 1]
        return csv_data

    def _call_scraper(self):
        USER_MAIL = os.getenv("TWITTER_MAIL")
        USER_UNAME = os.getenv("TWITTER_USERNAME")
        USER_PASSWORD = os.getenv("TWITTER_PASSWORD")

        scraper = Twitter_Scraper(
            mail=USER_MAIL,
            username=USER_UNAME,
            password=USER_PASSWORD,
        )
        scraper.login()
        scraper.scrape_tweets(
            scrape_query="heybanco (from:AngelGa35156959)",
        )
        scraper.save_to_csv()
        if not scraper.interrupted:
            scraper.driver.close()

    def _read_tweets_from_csv(self):
        tweets_folder_path = os.path.join(os.getcwd(), "tweets")
        for file in os.listdir(tweets_folder_path):
            if file.endswith(".csv"):
                csv_data = self._read_csv_to_string(os.path.join(tweets_folder_path, file))
                csv_data = self._clean_csv_data_string(csv_data)    
                for row in csv_data:
                    print("Row", row)
                    tweet_id = row[-1]
                    if tweet_id not in self.database_data:
                        timestamp = row[2].split("T")
                        date = timestamp[0]
                        time = timestamp[1]
                        tweet_text = row[4]
                        self.tweets_to_read[tweet_id] = {'date': date, 'time': time, 'tweet_text': tweet_text}
                os.remove(os.path.join(tweets_folder_path, file))
        return self.database_data

    def _dummy_evaluate(self, tweet):
        return random.choice([0, 1, 2])

    def _get_actual_tweets(self):
        self._call_scraper()
        self._read_tweets_from_csv()

    def _post_tweet_response(self, evaluation, tweet_id):
        response = self.TWEET_CONTENTS[evaluation]
        self.tweepy_client.create_tweet(text=response, in_reply_to_tweet_id=tweet_id)

    def _evaluate_tweets(self):
        for tweet_id in self.tweets_to_read:
            tweet_data = self.tweets_to_read[tweet_id]
            tweet_text = tweet_data['tweet_text']
            tweet_evaluation = self._dummy_evaluate(tweet_text)
            print("Tweet evaluation:", tweet_evaluation)
            print("Tweet ID:", tweet_id)
            if tweet_evaluation != 0:
                self._post_tweet_response(tweet_evaluation, tweet_id)
            self.database_data[tweet_id] = { 'date': tweet_data['date'], 'time': tweet_data['time'], 'tweet_text': tweet_text, 'evaluation': tweet_evaluation}
        self.tweets_to_read = {}
        return self.database_data

    def run_bot(self):
        self._get_actual_tweets()
        self._evaluate_tweets()
        #time.sleep(self.sleep_time)

if __name__ == "__main__":
    bot = Bot()
    bot.run_bot()