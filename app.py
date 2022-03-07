import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()  # convert to lower case
    text = nltk.word_tokenize(text)  # tokenization

    y = []
    for i in text:
        if i.isalnum():  # removing special characters fro the sentence
            y.append(i)

    text = y[:]  # list is mutable, to copy list we do cloning
    y.clear()

    for i in text:  # Removing stopwords and punctuation marks
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Stemming (converting similar words to a single word)

    return " ".join(y)

model = pickle.load(open('model.pkl', 'rb'))
tfidf  = pickle.load(open('vectorizer.pkl', 'rb'))


st.title('Sms Spam Detector')
st.subheader("Enter your message")
input_message = st.text_area(':')
if st.button("predict message"):


    transform_message = transform_text(input_message)

    vector_input = tfidf.transform([transform_message])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam Message")
    else:
        st.header("Not a spam")



