# Must use Python 3.5 to 3.8 to avoid unnecessary errors
import json
import pickle
import random
import numpy

# Library required by flask
from flask import Flask, render_template, request

# Libraries from NLTK to reduce the words to its stem, so that we don't lose any performance
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

# To avoid tensorflow print messages on standard error
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=numpy.VisibleDeprecationWarning)

# To load all_intents.json file for Chatty to pick up intents from the file
lemmatizer = WordNetLemmatizer()
all_intents = json.loads(open('all_intents.json').read())

# To load all_words.json pkl for Chatty to pick up words from the file
all_words = pickle.load(open('all_words.pkl', 'rb'))
# To load all_categories.pkl file for Chatty to pick up categories from the file
all_categories = pickle.load(open('all_categories.pkl', 'rb'))
# To load training_model.h5 file for Chatty to adopt the training_model
training_model = load_model('training_model.h5')

# Function: To apply tokenizer to the messages
def tokenize_the_message(messages):
    messages = nltk.word_tokenize(messages)
    return messages

# Function: To apply lemmatizer to the messages
def lemmatize_the_message(messages):
    messages = [lemmatizer.lemmatize(word.lower()) for word in messages]
    return messages

# Function: To convert messages into list of 0s and 1s, so the chatbot can know if the word are there or not
def bag_of_words(messages):
    messages = tokenize_the_message(messages)
    messages = lemmatize_the_message(messages)
    # To create a bag with zeros, there will be as many zeros as individual words
    initial_bag = [0] * len(all_words)
    for text in messages:
        for i, word in enumerate(all_words):
            if word == text:
                initial_bag[i] = 1
    return numpy.array(initial_bag)

# Function: TO predict categories of the messages
def predict_categories(messages):
    # To create a bag of words
    initial_bag = bag_of_words(messages)
    # To predict the outcomes based on those bag of words
    predict_outcomes = training_model.predict(numpy.array([initial_bag]))[0]
    # To allow for a certain percentages of uncertainty
    error_threshold = 0.25
    # Only if the outcome in the index are larger than the error_threshold
    predict_outcomes = [[index, predict_outcome] for index, predict_outcome in enumerate(predict_outcomes) if predict_outcome > error_threshold]
    # To take first index every time and reverse equals to true, to sort it reverse order, descending order essentially
    predict_outcomes.sort(key=lambda x: x[1], reverse=True)
    # To create an empty restore list for intents, classes & probabilities
    empty_restore_list = []
    for predict_outcome in predict_outcomes:
        empty_restore_list.append({"intent": all_categories[predict_outcome[0]], "probability": str(predict_outcome[1])})
    return empty_restore_list

# Function: TO get response from all_intents,json file depending on the intents_categories
def get_response(intents_categories, all_intents_json):
    global response_outcome
    input_tag = intents_categories[0]['intent']
    intents_contents = all_intents_json['intents']
    for index in intents_contents:
        # To get a random response from all_intents.json if the tag are same as the tag in the all_intents.json file
        if index['tag'] == input_tag:
            response_outcome = random.choice(index['responses'])
            break
    return response_outcome

# Function: for chatbot to pick a response
def chatbot_response(user_message):
    r = predict_categories(user_message)
    result = get_response(r, all_intents)
    return result

# Required by flask, to define the location for static files
app = Flask(__name__)
app.static_folder = 'static'
app.config.update()

# Required by flask, to define the template/main html page for the website
@app.route("/")
def home():
    return render_template("main.html")

# Required by flask, when user input a message, the chatbot will respond to it
@app.route("/get")
def get_bot_response():
    get_user_input = request.args.get('chatbotMessage')
    return chatbot_response(get_user_input)

# Required by flask, to enable the flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    
# Code below are for console debug only
# To print a short message for user to know what to do
# print("Enter your questions below :)")
# To loop this function over and over again
# while True:
    # To store user input into a temporary variable called message
    # message = input("")
    # To validate the user input, the following lines will only be activated when the user input are more than 1 letters
    # if len(message) > 1:
        # intents = predict_categories(message)
        # final_outcome = get_response(intents, all_intents)
        # print(final_outcome)
    # To validate the user input, if user input are less than 1 letter then the chatbot will display following message
    # else:
        # print("Sorry i dont understand... please try again")
