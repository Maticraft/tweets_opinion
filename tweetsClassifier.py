import torch
from nltk.stem import PorterStemmer
from utils import load_model, load_vocab, encode

# Classifier - model that predicts which class (positive or negative) represents the given tweet
# It uses the pretrained LSTM model (see lstm.py and LSTM_training.ipynb)

class Classifier():
    def __init__(self, vocab_path, model_path):
        self.lstm = load_model(model_path)
        self.vocab = load_vocab(vocab_path)
        self.ps = PorterStemmer()

    def predict(self, tweet, threshold = 0.5):
        tensor = encode(tweet, self.vocab, self.ps, self.lstm.seq_size)
        output = self.lstm(torch.unsqueeze(tensor, dim=0))

        if output.item() <= threshold:
            prediction = 0
        elif output.item() >= threshold:
            prediction = 1
        else:
            prediction = -1

        return prediction











