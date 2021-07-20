from string import punctuation
from nltk.stem import PorterStemmer

# parsing the file into separate words
def parse_file(filepath):  
    with open(filepath, 'r', encoding="latin-1") as file_obj:
        for line in file_obj:
            token = line.strip("\n").split('","')[5].translate(str.maketrans("", "", punctuation)).lower()
            words = token.split()
            yield from words

# saving the vocabulary
def save(filepath, words):
    with open(filepath, 'w', encoding = 'utf8') as file_obj:
        for word in words:
            file_obj.write(word + str("\n"))


# Constructing the vocabulary on the basis of N most common words in the data set file

N = 20000 
vocab = dict()
words = parse_file("tweets.csv")
ps = PorterStemmer()

for i, word in enumerate(words):
    if word.isnumeric():
        word = "numeric"
    else:
        word = ps.stem(word)

    if word in vocab.keys():
        vocab[word] += 1
    else: 
        vocab[word] = 1
    if i % 10000 == 0:
        print("Progress:", i)


vocablist = [(k, v) for k, v in vocab.items()]
vocabsorted = sorted(vocablist, key= lambda x: x[1], reverse= True)
Nwords = [k for k, v in vocabsorted[:N]]

save("./data/vocab.txt", Nwords)

