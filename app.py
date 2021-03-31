import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, render_template, url_for
import pickle

lem = WordNetLemmatizer()
stopwords_en = stopwords.words('english')
model = pickle.load(open("spam_ham.pkl", "rb"))
cv = pickle.load(open("cv.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/check', methods=['POST'])
def check():
    
    new_message = request.form['message']
    #print(new_message)
    new_length = len(new_message.split())
    new_message = re.sub('[^a-zA-Z]',' ', new_message)
    new_message= new_message.lower()
    new_message = nltk.word_tokenize(new_message)
    new_message = [lem.lemmatize(word) for word in new_message if not word in stopwords_en]
    #print(new_message)
    new_message = ' '.join(new_message)
    new_message = [new_message]
    new_countvector = cv.transform(new_message).toarray()

    new_df = pd.DataFrame(new_countvector, columns = np.arange(len(new_countvector[0])))
    new_df.insert(loc = 0, column = 'label', value = new_length)
    pred = model.predict(new_df)
    if pred[0]==0:
        prediction_text = 'Not a spam :)'
    else:
        prediction_text = "It's a SPAM!"
    return render_template('index.html', prediction_text='{}'.format(prediction_text))


if __name__ == "__main__":
    app.run(debug=True)