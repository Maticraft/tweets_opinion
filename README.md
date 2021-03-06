# tweets_opinion

Package consists of twitter api, which collects the stream of tweets with given filter (using tweepy) and the classifies each tweet's opinion as positive or negative (using the pretrained LSTM model). Subsequently, the histogram is plotted after each 100 of tweets. To use those scripts, please insert your own tokens in the twitterCredentials file. 

The LSTM model is implemented with pytorch and trained on the data from: https://www.kaggle.com/kazanova/sentiment140.


List of files:
- LSTM.ipynb - notebook used to train a network
- lstm.py - LSTM class implementation
- tweetsClassifier.py - model predicting the opinion of a tweet (based on the LSTM)
- twitterApi.py - main file
- utils.py - some functions
- vocab.py - script used to generate a vocabulary from the data set
- data/vocab.txt - vocabulary
- models/biLSTM.pt - trained LSTM model
- histograms - some examples of histograms generated by the api
- twitterCredentials_template.json - place your tokens here and rename to twitterCredentials.json
