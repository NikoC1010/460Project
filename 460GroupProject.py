import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from nltk.stem import PorterStemmer
import re
import string
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt_tab')

Corpus = pd.read_csv('/content/sample_data/train.csv', encoding='latin-1')

Corpus = Corpus[['Class Index', 'Description']]

Corpus.rename(columns={'Description': 'Data', 'Class Index': 'Label'}, inplace=True)

Corpus['Data'] = Corpus['Data'].fillna('') 

print(Corpus['Label'].dtype)
print(Corpus['Label'].value_counts())
print(Corpus['Label'].isnull().sum())

Corpus = Corpus[Corpus['Data'] != '']

samples1 = Corpus[Corpus['Label'] == 1].sample(n=5000)
samples2 = Corpus[Corpus['Label'] == 2].sample(n=5000)
samples3 = Corpus[Corpus['Label'] == 3].sample(n=5000)
samples4 = Corpus[Corpus['Label'] == 4].sample(n=5000)


Corpus = pd.concat([samples1, samples2, samples3, samples4])

Corpus = Corpus.sample(frac=1).reset_index(drop=True)

print(Corpus['Label'].value_counts())
print(Corpus.dtypes)

#Convert text to lowercase
Corpus['Data'] = [comment.lower() for comment in Corpus['Data']]

#Remove numbers
Corpus['Data'] = [re.sub(r'\d+', '', comment) for comment in Corpus['Data']]

#Remove punctuation
translator = str.maketrans('', '', string.punctuation)  # Create a translation table
Corpus['Data'] = [comment.translate(translator) for comment in Corpus['Data']]

#White spaces removal
Corpus['Data'] = [comment.strip() for comment in Corpus['Data']]

#Tokenization
Corpus['Data']= [word_tokenize(comment) for comment in Corpus['Data']]

# Remove stop words and apply lemmatizer
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
Corpus['Data'] = [
    [lemmatizer.lemmatize(token) for token in comment if token not in stop_words and token.isalpha()]
    for comment in Corpus['Data']
]

# Join tokens back into a single string for TfidfVectorizer
Corpus['Data'] = [' '.join(comment) for comment in Corpus['Data']]

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['Data'],Corpus['Label'],test_size=0.2)
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

pip install wordcloud matplotlib pandas

from wordcloud import WordCloud
import matplotlib.pyplot as plt

news = Corpus[Corpus['Label'] == 1]['Data']

news_text = ' '.join(news)

wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(news_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words for The World news')
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

news = Corpus[Corpus['Label'] == 2]['Data']

news_text = ' '.join(news)

wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(news_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words for Sport news')
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt


news = Corpus[Corpus['Label'] == 3]['Data']


news_text = ' '.join(news)


wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(news_text)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words for Business news')
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt


news = Corpus[Corpus['Label'] == 4]['Data']


news_text = ' '.join(news)


wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100, colormap='viridis').generate(news_text)


plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Words for Science/Technology news')
plt.show()

Tfidf_vector = TfidfVectorizer(max_features=10000)
Tfidf_vector.fit(Corpus['Data'])
Train_X_Tfidf = Tfidf_vector.transform(Train_X)
Test_X_Tfidf = Tfidf_vector.transform(Test_X)
print("Train_X_Tfidf shape:", Train_X_Tfidf.shape)
print("Test_X_Tfidf shape:", Test_X_Tfidf.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
Train_X_Tfidf = pca.fit_transform(Train_X_Tfidf.toarray())
Test_X_Tfidf = pca.transform(Test_X_Tfidf.toarray())
#print("Train_X_Tfidf shape:", Train_X_Tfidf.shape)
#print("Test_X_Tfidf shape:", Test_X_Tfidf.shape)

from scipy.sparse import csr_matrix

Train_X_Tfidf = csr_matrix(Train_X_Tfidf)
Test_X_Tfidf = csr_matrix(Test_X_Tfidf)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']

}

SVM = svm.SVC(decision_function_shape='ovo',probability=True)
grid_search = GridSearchCV(SVM, param_grid, scoring='roc_auc_ovo', cv=5, refit=True)
grid_search.fit(Train_X_Tfidf, Train_Y)
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("Best Parameters: ", best_params)

predictions_SVM1 = best_model.predict(Test_X_Tfidf)
predictions_SVM2 = best_model.predict(Train_X_Tfidf)

print("SVM Tesing Accuracy Score -> ",accuracy_score(predictions_SVM1, Test_Y)*100)
print("SVM Training Accuracy Score -> ",accuracy_score(predictions_SVM2, Train_Y)*100)

"""Model evaluation"""

from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

print("Testing Performance \n-------------------------------------------------------")

print(classification_report(Test_Y, predictions_SVM1, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))

print("\nTrianing Performance \n-------------------------------------------------------")

print(classification_report(Train_Y, predictions_SVM2, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

predictions_SVM1 = best_model.predict(Test_X_Tfidf)

cm = confusion_matrix(Test_Y, predictions_SVM1)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["The World", "Sports", "Business", "Science/Technology"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for 4-Class News Classification")
plt.show()