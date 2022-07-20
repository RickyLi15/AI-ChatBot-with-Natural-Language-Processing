import random 
import json
import pickle 
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras 
from keras.models import load_model
import webbrowser

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open("words.pkl", 'rb'))
classes = pickle.load(open("classes.pkl", 'rb'))
model = load_model('chatbotmodel.h5')

# function for cleaning up the sentences
def clean_up_sentence(sentence):
    # Converts to a list of words
    sentence_words = nltk.word_tokenize(sentence)
    # lemmatizes the words that are in sentence_words
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Converts a sentence into a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result, tag

print("Luc Bot is Running!!!")

# Creates the path and url that will allow python to open up google in chrome
url = 'https://www.google.com/'
chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
#Running the bot 
while True:
    message = input("")
    ints = predict_class(message)
    res,tag = get_response(ints, intents)
    print(res)
    # opens up chrome to google.com if the predicted tag is "GOOGlE"
    if tag == "Google":
        webbrowser.get(chrome_path).open(url)
