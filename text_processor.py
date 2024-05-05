#libraries
import pandas as pd
import re
import demoji
from deep_translator import GoogleTranslator

class TextProcessor:
    def __init__(self, file_path):
        """Initialize the TextProcessor with a CSV file path.

        Args:
            file_path (str): The path to the CSV file containing the text data.
        """
        self.data = pd.read_csv(file_path) 
        self.data.rename(columns = {"tweet":'text', "clase":"label"}, inplace=True)

    def clean_text(self, text):
        """Clean the text by converting it to lowercase and removing special characters.

        Args:
            text (str): The text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        cleaned_text = text.lower()
        cleaned_text = re.sub("[.,!?¡¿*()-\]}$_'“”&^è]","", cleaned_text)
        return cleaned_text

    def translate_emojis(self, text):
        
        """ Translate emojis to english words using demoji. 
            Translate english words to spanish using Google Translator

        Args:
            text (str): The text containing emojis to be translated.

        Returns:
            str: The text with emojis translated to Spanish.
        """
        if demoji.findall_list(text):
            emoji_translation = demoji.replace(text, repl = GoogleTranslator(source='english', target='spanish').translate((demoji.findall_list(text)[0])) )
            return emoji_translation
        else:
            return text

    def process_data(self):
        """Process the text data by cleaning text and translating emojis.

        Returns:
            pandas.DataFrame: The processed text data as a DataFrame.
        """
        self.data['text'] = self.data['text'].apply(self.clean_text)
        self.data['text'] = self.data['text'].apply(self.translate_emojis)
        new_data = self.data[['text', 'label']]
        return new_data

if __name__ == "__main__":
    file_path = 'dataset_1279.csv'
    processor = TextProcessor(file_path)
    new_data = processor.process_data()
    new_data.to_csv('cleaned_data.csv', index=False)