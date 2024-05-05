"""
A module containing a class representing a Twitter bot that reads and evaluates tweets, and posts responses.

Authors:
- Jose Angel Garcia Gomez
- Pablo Gonzalez de la Parra
- Jose Maria Ibarra Perez
- Ana Martinez Barbosa

Twitter Scrapper Module obtained from: https://github.com/godkingjay/selenium-twitter-scraper
"""

from predictor import BertModelWrapper
from selenium_scraper.scraper.twitter_scraper import Twitter_Scraper
import os
import random
import tweepy
import time
import csv

from dotenv import load_dotenv
load_dotenv()

class Bot:
    """
    A class representing a Twitter bot that reads and evaluates tweets, and posts responses.

    Attributes:
    - TWEET_CONTENTS (dict): A dictionary containing tweet response contents.
    - model_weight_path (str): Path to the model weights.
    - database_data (dict): Database containing tweet data.
    - sleep_time (int): Time interval between bot operations.
    - tweepy_client (tweepy.Client): Client for interacting with the Twitter API.
    """

    TWEET_CONTENTS = {
            1: ["¡Hola! Por favor, nos puedes contactar en el 81 4392 2626 para poder apoyarte a revisar la situación.",
            "¡Hola! En seguida te envíamos DM con mayor información al respecto. 🙌",
            "¡Hola, buen día! Por favor, nos puedes escribir al correo de contacto@heybanco.com, desde el correo registrado en tu cuenta para poder apoyarte."],
            2: ["¡Descubre Hey Banco, tu aliado digital para gestionar tus finanzas! Con nuestra tarjeta virtual, disfruta de compras en línea, meses sin intereses, recompensas y alertas de transacciones. Abre tu cuenta y disfruta de estos beneficios. ¡Haz clic para registrarte! https://quierosercliente.hey.inc",
                "No busques más. El banco para ti. Descubre Hey Banco, tu aliado digital para gestionar tus finanzas con nuestra tarjeta virtual. Disfruta de compras en línea, meses sin intereses, recompensas y alertas de transacciones. Abre tu cuenta y disfruta de estos beneficios. Haz clic para registrarte: https://quierosercliente.hey.inc/",
                "Si tu banco actual te decepcionó, estamos aquí para cubrirte. Atención al cliente excepcional en Hey Banco. Descubre nuestro aliado digital para gestionar tus finanzas. Disfruta de compras en línea, meses sin intereses, recompensas y alertas de transacciones. Abre tu cuenta y disfruta de estos beneficios. Haz clic para registrarte: https://quierosercliente.hey.inc"
                ]}

    def __init__(self, model_weight_path="", database_data={}, sleep_time=120):
        """
        Initialize the Bot object.

        Parameters:
        - model_weight_path (str): Path to the model weights.
        - database_data (dict): Database containing tweet data.
        - sleep_time (int): Time interval between bot operations.
        """
        self.model = self._dummy_evaluate
        if model_weight_path:
            model_weight_path = os.path.join(os.getcwd(), model_weight_path)
            bert_model = BertModelWrapper(model_weight_path)
            self.model = bert_model.predict
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
        """
        Read CSV data from file and convert it to a string.

        Parameters:
        - file_path (str): Path to the CSV file.

        Returns:
        - csv_data (str): CSV data as a string.
        """
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            csv_data = '\n'.join(','.join(row) for row in reader)
        return csv_data

    def _clean_csv_data_string(self, csv_data):
        """
        Clean CSV data string.

        Parameters:
        - csv_data (str): CSV data as a string.

        Returns:
        - cleaned_data (list): Cleaned CSV data as a list of lists.
        """
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
        """
        Call the Twitter scraper module to fetch tweets.
        """
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
        """
        Read tweets from CSV files and store them for evaluation.
        """
        tweets_folder_path = os.path.join(os.getcwd(), "tweets")
        for file in os.listdir(tweets_folder_path):
            if file.endswith(".csv"):
                csv_data = self._read_csv_to_string(os.path.join(tweets_folder_path, file))
                csv_data = self._clean_csv_data_string(csv_data)    
                for row in csv_data:
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
        """
        Dummy function to evaluate tweets.

        Parameters:
        - tweet (str): Text of the tweet.

        Returns:
        - evaluation (int): Evaluation result.
        """
        return random.choice([0, 1, 2])

    def _get_actual_tweets(self):
        """
        Fetch actual tweets using the scraper.
        """
        self._call_scraper()
        self._read_tweets_from_csv()

    def _post_tweet_response(self, evaluation, tweet_id):
        """
        Post response to a tweet.

        Parameters:
        - evaluation (int): Evaluation result.
        - tweet_id (str): ID of the tweet.
        """
        response = random.choice(self.TWEET_CONTENTS[evaluation])
        try:
            self.tweepy_client.create_tweet(text=response, in_reply_to_tweet_id=tweet_id)
        except Exception as e:
            print("Error posting tweet response:", e)

    def _evaluate_tweets(self):
        """
        Evaluate tweets and post responses.
        """
        for tweet_id in self.tweets_to_read:
            tweet_data = self.tweets_to_read[tweet_id]
            tweet_text = tweet_data['tweet_text']
            tweet_evaluation = self.model(tweet_text)
            print("Tweet evaluation:", tweet_evaluation)
            print("Tweet ID:", tweet_id)
            if tweet_evaluation != 0:
                self._post_tweet_response(tweet_evaluation, tweet_id)
            self.database_data[tweet_id] = { 'date': tweet_data['date'], 'time': tweet_data['time'], 'tweet_text': tweet_text, 'evaluation': tweet_evaluation}
        self.tweets_to_read = {}
        return self.database_data

    def run_bot(self):
        """
        Run the Twitter bot to fetch, evaluate, and respond to tweets.
        """
        while True:
            self._get_actual_tweets()
            self._evaluate_tweets()
            time.sleep(self.sleep_time)

if __name__ == "__main__":
    bot = Bot(model_weight_path="model/heybot_model.pth")
    bot.run_bot()