import pandas as pd
import re
import streamlit as st
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

news_df = pd.read_csv('train3.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label2', axis=1)
y = news_df['label2']

ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_df['content'] = news_df['author']+" "+news_df['title']

X = news_df['content'].values
y = news_df['label2'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
#new
#y = y.astype(int)

class_counts = np.bincount(y)
classes_with_enough_samples = np.where(class_counts >= 2)[0]
filtered_indices = [i for i, label2 in enumerate(y) if label2 in classes_with_enough_samples]

X = X[filtered_indices]
y = y[filtered_indices]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

model = LogisticRegression()
model.fit(X_train,Y_train)

#web
st.title('Sem Fake')
input_text = st.text_input('Adicione o texto de busca', autocomplete='off')

def prediction(input_text, vectorizer, classifier):
    input_data = vectorizer.transform([input_text])
    prediction = classifier.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text, vector, model)
    if pred == 1:
        st.write('É Fato')
    else:
        st.write('É Fake')
