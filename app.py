import streamlit as st
import pickle
from  nltk import word_tokenize as wt

from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')
st.title('SMS Spam Detector')
sms = st.text_area('Enter The Message')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
#preprocess
ps=PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
  text = y[:]

  y.clear()
  for i in text:
    if (i not in stopwords.words('english')) and (i not in string.punctuation)  and (len(i)!=1) and (i.isdigit()==False):
      y.append(i)
  text = y[:]

  y.clear()
  for i in text:
    y.append(ps.stem(i))
  return ' '.join(y)

if st.button('Detect'):

    trans_sms=transform_text(sms)
    #vectorize
    vector = tfidf.transform([trans_sms])
    #predict
    pred=model.predict(vector)
    #display
    if pred == 1:
        st.markdown('<h2 style="color:red">Spam</h2>',unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color:green">Not Spam</h2>',unsafe_allow_html=True)
        


