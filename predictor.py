import torch
from transformers import BertForSequenceClassification,BertTokenizer

class BertModelWrapper:
    def __init__(self, model_path):
        self.model = BertForSequenceClassification.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-uncased',
            num_labels=3,
            output_attentions=False,
            output_hidden_states=False,
        )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, new_sentence):
        encoding = self.preprocess(new_sentence)
        with torch.no_grad():
            output = self.model(encoding['input_ids'], token_type_ids=None, attention_mask=encoding['attention_mask'])[0]
        prediction = torch.argmax(output).item()
        return prediction

    def preprocess(self, sentence):
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
