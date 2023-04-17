import re
import contractions
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
nltk.download('words') #download list of english words
nltk.download('stopwords') #download list of stopwords
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
englishWords = set(nltk.corpus.words.words())
from nltk.tokenize import RegexpTokenizer
import pandas as pd
tokenizer = RegexpTokenizer(r'\w+')


def remove_stopWords(tokens):
    return [w for w in tokens if (w in englishWords and w not in stopWords)]

# Convert the nltk pos tags to tags that wordnet can recognize
def nltkToWordnet(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None
    
# Lemmatize a list of words/tokens
def lemmatize(tokens):
  pos_tags = nltk.pos_tag(tokens)    
  res_words = []
  for word, tag in pos_tags:
    tag = nltkToWordnet(tag)    
    if tag is None:                        
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))
  return res_words

def removeUrl(text):
    text = re.sub(r'http\S+', '', text.lower())
    return text

def remove_contractions(text): 
    return ' '.join([contractions.fix(word) for word in text.split()])

def startPreProcessing(dataframe):
    for data in dataframe.text:
        text = data
        text = re.sub(r'http\S+', '', text.lower())
        text = ' '.join([contractions.fix(word) for word in text.split()])
        tokens = tokenizer.tokenize(text)
        tokens = lemmatize(tokens)
        tokens = remove_stopWords(tokens)
        data = ' '.join(tokens)
    return dataframe

df = pd.read_csv('dataset/fake-and-real-news-dataset/Fake.csv')[:3]
df2 = startPreProcessing(df)

print(df)
print(df2)

