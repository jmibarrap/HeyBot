"""
Trainer module for training a BERT-based sequence classification model.

Authors:
- Jose Angel Garcia Gomez
- Pablo Gonzalez de la Parra
- Jose Maria Ibarra Perez
- Ana Martinez Barbosa

"""

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import trange
import numpy as np

class Processor:
    """
    Processor class for preparing data for model training.

    Args:
        tokenizer (transformers.BertTokenizer): Tokenizer object for tokenizing input texts.
        max_length (int): Maximum length of tokenized sequences.
        batch_size (int): Batch size for DataLoader objects.
        val_ratio (float, optional): Ratio of validation data split. Defaults to 0.2.
        random_state (int, optional): Random state for data splitting. Defaults to 42.
    """
    def __init__(self, tokenizer, max_length, batch_size, val_ratio=0.2, random_state=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.random_state = random_state

    def prepare_data(self, texts, labels):
        """
        Prepares data for training.

        Tokenizes input texts, splits data into train and validation sets,
        and creates TensorDataset objects for each set.

        Args:
            texts (numpy.ndarray): Array of input texts.
            labels (numpy.ndarray): Array of corresponding labels.

        Returns:
            tuple: (train_set, val_set) Tuple of TensorDataset objects for training and validation sets.
        """
        tokenized_texts = [self.tokenizer.encode_plus(
                                text,
                                add_special_tokens=True,
                                max_length=self.max_length,
                                pad_to_max_length=True,
                                return_attention_mask=True,
                                return_tensors='pt'
                            ) for text in texts]

        token_ids = torch.cat([encoding['input_ids'] for encoding in tokenized_texts], dim=0)
        attention_masks = torch.cat([encoding['attention_mask'] for encoding in tokenized_texts], dim=0)
        labels_tensor = torch.tensor(labels)

        train_idx, val_idx = train_test_split(
            np.arange(len(labels)),
            test_size=self.val_ratio,
            shuffle=True,
            stratify=labels
        )

        train_set = TensorDataset(token_ids[train_idx], attention_masks[train_idx], labels_tensor[train_idx])
        val_set = TensorDataset(token_ids[val_idx], attention_masks[val_idx], labels_tensor[val_idx])

        return train_set, val_set

    def get_train_dataloader(self, train_set):
        """
        Creates a DataLoader object for the training set.

        Args:
            train_set (torch.utils.data.dataset.TensorDataset): TensorDataset object for the training set.

        Returns:
            torch.utils.data.dataloader.DataLoader: DataLoader object for the training set.
        """
        train_dataloader = DataLoader(
            train_set,
            sampler=RandomSampler(train_set),
            batch_size=self.batch_size
        )
        return train_dataloader

    def get_val_dataloader(self, val_set):
        """
        Creates a DataLoader object for the validation set.

        Args:
            val_set (torch.utils.data.dataset.TensorDataset): TensorDataset object for the validation set.

        Returns:
            torch.utils.data.dataloader.DataLoader: DataLoader object for the validation set.
        """
        val_dataloader = DataLoader(
            val_set,
            sampler=SequentialSampler(val_set),
            batch_size=self.batch_size
        )
        return val_dataloader


class Model:
    """
    Model class for training and evaluation.

    Args:
        model (transformers.BertForSequenceClassification): BERT-based sequence classification model.
        train_dataloader (torch.utils.data.dataloader.DataLoader): DataLoader object for the training set.
        val_dataloader (torch.utils.data.dataloader.DataLoader): DataLoader object for the validation set.
        optimizer (torch.optim.optimizer.Optimizer): Optimizer for model training.
        epochs (int, optional): Number of training epochs. Defaults to 25.
    """
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, epochs=25):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.epochs = epochs

    def b_tp(self, preds, labels):
        """
        Returns the count of true positives.

        Args:
            preds (numpy.ndarray): Predicted labels.
            labels (numpy.ndarray): True labels.

        Returns:
            int: Count of true positives.
        """
        return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

    def b_fp(self, preds, labels):
        """
        Returns the count of false positives.

        Args:
            preds (numpy.ndarray): Predicted labels.
            labels (numpy.ndarray): True labels.

        Returns:
            int: Count of false positives.
        """
        return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

    def b_tn(self, preds, labels):
        """
        Returns the count of true negatives.

        Args:
            preds (numpy.ndarray): Predicted labels.
            labels (numpy.ndarray): True labels.

        Returns:
            int: Count of true negatives.
        """
        return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

    def b_fn(self, preds, labels):
        """
        Returns the count of false negatives.

        Args:
            preds (numpy.ndarray): Predicted labels.
            labels (numpy.ndarray): True labels.

        Returns:
            int: Count of false negatives.
        """
        return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

    def b_metrics(self, preds, labels):
        """
        Calculates classification metrics including accuracy, precision, recall, and specificity.

        Args:
            preds (numpy.ndarray): Predicted labels.
            labels (numpy.ndarray): True labels.

        Returns:
            tuple: (accuracy, precision, recall, specificity) Classification metrics.
        """
        preds = np.argmax(preds, axis=1).flatten()
        labels = labels.flatten()
        tp = self.b_tp(preds, labels)
        tn = self.b_tn(preds, labels)
        fp = self.b_fp(preds, labels)
        fn = self.b_fn(preds, labels)
        b_accuracy = (tp + tn) / len(labels)
        b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
        b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
        b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
        return b_accuracy, b_precision, b_recall, b_specificity

    def train(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Trains the model for the specified number of epochs and evaluates performance on the validation set.

        Args:
            device (torch.device, optional): Device to perform training. Defaults to GPU if available, otherwise CPU.
        """
        for epoch in trange(self.epochs, desc='Epoch'):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(self.train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.optimizer.zero_grad()
                train_output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                train_output.loss.backward()
                self.optimizer.step()
                tr_loss += train_output.loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            self.model.eval()
            val_accuracy = []
            val_precision = []
            val_recall = []
            val_specificity = []

            for batch in self.val_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    eval_output = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = eval_output.logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                b_accuracy, b_precision, b_recall, b_specificity = self.b_metrics(logits, label_ids)
                val_accuracy.append(b_accuracy)
                if b_precision != 'nan': val_precision.append(b_precision)
                if b_recall != 'nan': val_recall.append(b_recall)
                if b_specificity != 'nan': val_specificity.append(b_specificity)

            print('\n\t - Train loss: {:.4f}'.format(tr_loss / nb_tr_steps))
            print('\t - Validation Accuracy: {:.4f}'.format(sum(val_accuracy) / len(val_accuracy)))
            print('\t - Validation Precision: {:.4f}'.format(sum(val_precision) / len(val_precision)) if len(
                val_precision) > 0 else '\t - Validation Precision: NaN')
            print('\t - Validation Recall: {:.4f}'.format(sum(val_recall) / len(val_recall)) if len(
                val_recall) > 0 else '\t - Validation Recall: NaN')
            print('\t - Validation Specificity: {:.4f}\n'.format(sum(val_specificity) / len(val_specificity)) if len(
                val_specificity) > 0 else '\t - Validation Specificity: NaN')

    def predict_class(self, sentence, tokenizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Predicts the class of a new sentence using the trained model.

        Args:
            sentence (str): Input sentence.
            tokenizer (transformers.BertTokenizer): Tokenizer object for tokenizing the sentence.
            device (torch.device, optional): Device to perform inference. Defaults to GPU if available, otherwise CPU.

        Returns:
            int: Predicted class label.
        """
        test_ids = []
        test_attention_mask = []

        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=32,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        test_ids.append(encoding['input_ids'])
        test_attention_mask.append(encoding['attention_mask'])
        test_ids = torch.cat(test_ids, dim=0)
        test_attention_mask = torch.cat(test_attention_mask, dim=0)

        with torch.no_grad():
            output = self.model(test_ids.to(device), token_type_ids=None, attention_mask=test_attention_mask.to(device))

        prediction = np.argmax(output.logits.cpu().numpy()).flatten().item()

        print('Input Sentence: ', sentence)
        print('Predicted Class: ', prediction)
        return prediction

if __name__ == '__main__':
    model = BertForSequenceClassification.from_pretrained(
        'dccuchile/bert-base-spanish-wwm-uncased',
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False,
    )

    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', do_lower_case=True)
    processor = Processor(tokenizer=tokenizer, max_length=32, batch_size=16, val_ratio=0.2)
    df = pd.read_csv('E:\data\hey_banco.csv')
    df = df[['text', 'label']]
    texts = df['text'].values
    labels = df['label'].values
    train_set, val_set = processor.prepare_data(texts, labels)
    train_dataloader = processor.get_train_dataloader(train_set)
    val_dataloader = processor.get_val_dataloader(val_set)

    optimizer = torch.optim.AdamW(model.parameters(),
                                lr = 5e-5,
                                eps = 1e-08
                                )

    model_trainer = Model(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, optimizer=optimizer)
    model_trainer.train()
    torch.save(model.state_dict(), 'heybot_model.pth')
