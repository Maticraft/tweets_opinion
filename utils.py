import torch
import json
from lstm import LSTM
from string import punctuation

# loading the pretrained LSTM model
def load_model(filepath):
    batch_size = 256
    wordsNb = 30
    hidden_size = 128
    classes = 1
    num_layers = 1
    input_size = 20002

    lstm = LSTM(input_size, hidden_size, num_layers, wordsNb, classes, batch_size)
    state = torch.load(filepath, map_location=torch.device('cpu'))
    lstm.load_state_dict(state)
    lstm.eval()

    return lstm

# loading the vocabulary
def load_vocab(filepath):
  vocab = []
  with open(filepath, encoding='utf8') as file_obj:
    for line in file_obj:
      token = line.strip("\n")
      vocab.append(token)

  return vocab


# encoding a token (tweet text) as a torch tensor
def encode(token, vocab, ps, length):
    items = token.lower().translate(str.maketrans("", "", punctuation)).split()
    tensor = torch.zeros(length, len(vocab) + 2)

    for i, item in enumerate(items):
        if i < length:
            if item.isnumeric():
                item = "numeric"
            else:
                item = ps.stem(item)

            try:
                indx = vocab.index(item)
            except ValueError:
                indx = len(vocab)

            tensor[i][indx] = 1

    if len(items) != 0 and i < (length - 1):
        indx = len(vocab) + 1

        for j in range(i + 1, length):
            tensor[j][indx] = 1

    if len(items) == 0:
        indx = len(vocab) + 1

        for j in range(0, length):
            tensor[j][indx] = 1

    return tensor


# loading the credentials needed for the twitter authentication
def credentials(filename):
    with open(filename, "r+") as config_file:
        config = json.loads(config_file.read())
    return config

