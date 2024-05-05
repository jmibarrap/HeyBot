"""
A module containing a class representing a simple memory based database.

Authors:
- Jose Angel Garcia Gomez
- Pablo Gonzalez de la Parra
- Jose Maria Ibarra Perez
- Ana Martinez Barbosa

"""

import torch
from transformers import BertForSequenceClassification,BertTokenizer

class BertModelWrapper:
    """
    A wrapper class for a BERT model for sequence classification.

    Attributes:
    - model (BertForSequenceClassification): BERT model for sequence classification.
    """

    def __init__(self, model_path):
        """
        Initialize the BertModelWrapper object.

        Parameters:
        - model_path (str): Path to the BERT model weights.
        """
        self.model = BertForSequenceClassification.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-uncased',
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, new_sentence):
        """
        Predict the class label for a new sentence.

        Parameters:
        - new_sentence (str): New sentence for classification.

        Returns:
        - prediction (int): Predicted class label.
        """
        encoding = self.preprocess(new_sentence)
        with torch.no_grad():
            output = self.model(encoding['input_ids'], token_type_ids=None, attention_mask=encoding['attention_mask'])[0]
        prediction = torch.argmax(output).item()
        return prediction

    def preprocess(self, sentence):
        """
        Preprocess a sentence for input to the model.

        Parameters:
        - sentence (str): Input sentence to preprocess.

        Returns:
        - inputs (dict): Preprocessed inputs for the model.
        """
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
        inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return inputs

if __name__ == "__main__":
    model_path = 'heybot_model.pth'
    bert_model = BertModelWrapper(model_path)
    new_sentence = 'amo hey banco pero estoy insatisfecha con el servicio'
    prediction = bert_model.predict(new_sentence)
    print('Input Sentence:', new_sentence)
    print('Predicted Class:', prediction)
