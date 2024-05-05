# from Model import Model
# from Preprocessing import Preprocessing
from selenium_scraper.scraper.twitter_scraper import Twitter_Scraper
import os
import sys

from dotenv import load_dotenv
load_dotenv()

class Bot:

    def __init__(self, model_weight_path="", database_data={}):
        #self.model = Model(model_weight_path) 
        #self.preprocessing = Preprocessing()
        self.database_data = database_data
        self.tweets_to_read = {}
    
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

    def monitor_tweets(self):
        self._call_scraper(tweets=10, query="heybanco (from:AngelGa35156959)")
        self._read_tweets_from_csv()
        print(self.tweets_to_read) # Debugging purposes

    
    def _read_tweets_from_csv(self):
        tweets_folder_path = os.path.join(os.getcwd(), "tweets")
        for file in os.listdir(tweets_folder_path):
            if file.endswith(".csv"):
                with open(os.path.join(tweets_folder_path, file), "r") as f:
                    file_data = f.readlines()
                    for line in file_data:
                        column_data = line.split(",")
                        tweet_id = column_data[-1]
                        timestamp = column_data[2].split("t")
                        date = timestamp[0]
                        time = timestamp[1]
                        tweet_text = column_data[4]
                        self.tweets_to_read[tweet_id] = { 'date': date, 'time': time, 'tweet_text': tweet_text}
                #os.remove(os.path.join(tweets_folder_path, file))
        return self.database_data
    
if __name__ == "__main__":
    bot = Bot()
    bot.monitor_tweets()