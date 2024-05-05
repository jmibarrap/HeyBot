# HeyBot

## Description

HeyBot is a Twitter bot designed to interact with users' tweets, evaluate sentiment, and respond accordingly. It utilizes various components such as scraping tweets, processing text data, and employing a machine learning model for sentiment analysis.

## Components

1. **Twitter Scraper:**

   - A module for scraping tweets from Twitter using Selenium.

2. **Text Processing:**

   - Cleans and processes text data, including translating emojis, using the Deep Translator library and demoji.

3. **Bert Model for Sentiment Analysis:**

   - Utilizes a pre-trained BERT model for sentiment analysis on Spanish text.

4. **Database Management:**

   - Manages a simple memory-based database for storing tweet information.

5. **Model Training and Evaluation:**

   - Trains the sentiment analysis model on labeled data and evaluates its performance using classification metrics.

6. **Selenium Scraper:**

   - Scrapes tweets from Twitter using Selenium.

7. **Streamlit dashboard**

   - Streamlit implementation that allows prediction monitoring and description.

8. **Training Notebook**

   - Allows training for evaluation with a jupyter notebook, homologous to training scripts used.

**Files:**

- `bot.py`: Main file containing the Twitter bot implementation.
- `database.py`: Database management class.
- `bert_model.py`: Wrapper for the BERT model used for sentiment analysis.
- `text_processor.py`: Text processing class for cleaning and translating text data.
- `processor.py`: Prepares data for model training.
- `model.py`: Handles model training and evaluation.

## Reusability and Attribution

The code inside the `selenium_scraper` folder is reusable and is integrated into various projects requiring Twitter data scraping.
Since it's part of a larger project, the code was inspired by a project found on GitHub. Proper attribution goes to the original author, @godkingjay.

Link to the original project: [selenium-twitter-scraper](https://github.com/godkingjay/selenium-twitter-scraper)

Also, the construction of the model was inspired by the following article:

Link to the article: [Fine-tuning BERT for Text Classification](https://towardsdatascience.com/fine-tuning-bert-for-text-classification-54e7df642894#ec34)


## Usage

1. Clone the repository:

   ```
   git clone https://github.com/jmibarrap/HeyBot.git
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file and add your Twitter API credentials.
   - Follow the format shown below:
      ```
      TWITTER_USERNAME=
      TWITTER_PASSWORD=
      TWITTER_MAIL=
      TWITTER_API_KEY=
      TWITTER_API_SECRET=
      TWITTER_ACCESS_TOKEN=
      TWITTER_ACCESS_TOKEN_SECRET=
      ```
   - Ensure you have the necessary environment variables set for Twitter login and API access.

4. Run the bot:
   ```
   python bot.py
   ```

## Model performance

	 - Train loss: 0.0306
	 - Validation Accuracy: 0.4702
	 - Validation Precision: 0.7406
	 - Validation Recall: 0.7477
	 - Validation Specificity: 0.8691

    On 80/20 split, end-to-end fine tuned for 25 epochs with the obtained extended dataset consisting of 1617 entries. 

**Dataset:**

- `dataset_2033.csv`: Example dataset for sentiment analysis training.

**Pre-trained Model:**

- `heybot_model.pth`: Pre-trained BERT model for sentiment analysis.

**Notes:**

- Ensure you have the necessary environment variables set for Twitter API access and other configurations.
- Modify the code as needed for your specific use case, such as changing the dataset, adjusting model parameters, or adding new functionality.
