import tweepy
import numpy as np
import matplotlib.pyplot as plt
from tweetsClassifier import Classifier
from utils import credentials

# Twitter api that collects a stream of tweets with given filters and classifies them as positive or negative
# After each 100 of tweets it plots a histogram of all collected tweets 
class StreamListener(tweepy.StreamListener):
    def __init__(self, model_path, vocab_path, filters = [], max_tweets = 10000, threshold = 0.5):
        super(StreamListener, self).__init__()
        self.classifier = Classifier(vocab_path, model_path)
        self.labels = []
        self.max_tweets = max_tweets
        self.timestamp = 0
        self.threshold = threshold
        self.filters = ", ".join(filters)

    def on_status(self, status):
        token = status.text
        label = self.classifier.predict(token, self.threshold)

        if len(self.labels) < self.max_tweets:
            self.labels.append(label)
        else:
            self.labels.pop(0)
            self.labels.append(label)
        
        self.timestamp += 1
        if self.timestamp % 100 == 0:
            self.plot_hist()

        print(token, label)


    def plot_hist(self):
        fig, ax = plt.subplots()
        ax.hist(self.labels, bins= [-0.5, 0.5, 1.5], rwidth=0.5)
        fig.suptitle('Opinion associated with words: ' + self.filters)
        ax.set_xlabel('Opinion')
        ax.set_ylabel('Number of tweets')
        xlabels = ['Negative', 'Positive']
        ax.set_xticks([0, 1])
        ax.set_xticklabels(xlabels)
        plt.savefig(f"./histograms/opinion_[{self.filters}]_t{self.timestamp}.png")



def tweets_stream(consumer_key, consumer_secret, access_token, access_token_secret, filters):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    twitter_listener = StreamListener("./models/biLSTM.pt", "./data/vocab.txt", filters, threshold = 0.5)
    tweets_stream = tweepy.Stream(auth = api.auth, listener = twitter_listener)
    tweets_stream.filter(languages = ['en'], track = filters)

    return twitter_listener, tweets_stream


auth_config = credentials("twitterCredentials.json")
tl, ts = tweets_stream(**auth_config, filters = ["vaccine"])