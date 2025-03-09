# Importing the dependencies
from string import punctuation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re   # Used to remove certain patterns from the texts
import string   # Used to get a list of all the punctuation in english language.
from textblob import TextBlob   # Used to handle spelling mistakes in the data.
import nltk   # Used for creating word vectors and text cleaning.
from nltk.tokenize import word_tokenize, sent_tokenize   # Used for tokenization.
from nltk.stem.porter import PorterStemmer   # Used for stemming (converting words into their root words)
from nltk.stem import WordNetLemmatizer   # Used for lemmatization
import emoji   # This library helps us to convert emojis into simple statements. For eg : smile emoji will be converted into :smile: .
from gensim.models import Word2Vec   # Used to convert the words into vectors.
import gensim.downloader as api
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('punkt')

wv = api.load('word2vec-google-news-300')   # Downloading the pretrained gensim model.

data = pd.read_csv("../Datasets/IMDB_Dataset.csv")

# Getting insights about the data
print(f"Shape of the data {data.shape}")
print(data.info())
sns.displot(data['sentiment'])  # Plotting the target column to visualise if there are any imbalances.
plt.show()
print(data.isnull().sum())   # Checking for null values, if any.

# Conclusion :: The data don't have any imbalance and null values

# Mapping the target column
data['sentiment'] = data['sentiment'].map({'positive' : 1, 'negative' : 0})
# positive --> 1
# negative --> 0

# Converting all the characters to lower case to fix the issue of getting different values for upper and lower cases.
data['review'] = data['review'].map(lambda x : x.lower())

# Writing a function to do all the preprocessing
def preprocess(text : str):
    """
    This function takes string a perform html tag removal, lemmatization and stemming and converts it to word embeddings.
    """
    pattern = re.compile('<.*?>')
    without_pattern = pattern.sub(r'', text)  # Removed the html tags.

    punc = string.punctuation   # Getting the punctuations
    text_no_punctuation = without_pattern.translate(str.maketrans('', '', punc))   # Removed the punctuations.

    textBlb = TextBlob(text_no_punctuation)
    correct_text = textBlb.correct().string

    stopwords = nltk.corpus.stopwords.words('english')   # This contains all the stopwords in the english language.
    clean_text = []
    for word in text.split():
        if word in stopwords:
            clean_text.append('')
        else :
            clean_text.append(word)
    clean_text = ' '.join(clean_text)   # Removed the stopwords.

    clean_text = emoji.demojize(clean_text)   # Converting the emojis into plain text.

    #ps = PorterStemmer()
    #processed_text = ' '.join([ps.stem(word) for word in tokens])   # Performing stemming .

    # We will be using lemmatization instead of stemming because lemmatization returns meaningful word.
    # Converting the processed text into vectors.
    lem = WordNetLemmatizer()
    tokens = word_tokenize(clean_text)   # Converting the sentences into tokens.
    lem_tokens = [lem.lemmatize(word) for word in tokens]

    sentence_vector = wv.get_mean_vector(lem_tokens, pre_normalize = True)

    return sentence_vector

data['review'] = data['review'].apply(lambda x : preprocess(x))
print(data.head(2))

# Splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x = data['review'], y = data['sentiment'], stratify = data['sentiment'], test_size = 0.2, random_state = 21)

# Training two models
model1 = XGBClassifier()
model2 = SVC(kernel = 'linear')

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)

model1_predict = model1.predict(x_test)
model2_predict = model2.predict(x_test)

# Comparing the accuracy of both the models
print(f"Model1 classification report ::")
print(classification_report(y_true = y_test, y_pred = model1_predict))
print('-------------------------------------------------------------------------------')
print(f"Model2 classification report ::")
print(classification_report(y_true = y_test, y_pred = model2_predict))



