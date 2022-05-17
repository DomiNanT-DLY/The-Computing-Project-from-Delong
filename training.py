# Must use Python 3.5 to 3.8 for Tensorflow library to work
import json
import pickle
import random
import numpy as np

# Libraries from NLTK to reduce the all_words to its stem, so that we don't lose any performance
import nltk
from nltk.stem import WordNetLemmatizer

# Libraries from Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import adam_v2

# To avoid tensorflow print on standard error
import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# To load all_intents.json file for training model
lemmatizer = WordNetLemmatizer()
all_intents = json.loads(open('all_intents.json').read())

all_words = []
all_categories = []
all_combinations = []

# These common punctuations the chatbot will not take care of
ignore_punctuation_list = [',', '.', '?', '!', ':', ';', '(', ')', '@']

# Intent is the object and intents is the sub values within the object
for intent in all_intents['intents']:
    # Pattern is the pattern list with in the object
    for pattern in intent['patterns']:
        # Tokenize is to split up the messages into individual words
        word_list = nltk.word_tokenize(pattern)
        # When the messages has been tokenized, all the words will be added into the words list
        all_words.extend(word_list)
        # To documents the words list into categories/tag
        all_combinations.append((word_list, intent['tag']))
        # To Check if the categories is already in the categories list or not
        if intent['tag'] not in all_categories:
            all_categories.append((intent['tag']))
# Testing to see if tokenize function is working properly or not
# print(all_categories)

# To sort & lemmatize words in the words list, so that we can reduce a given word to its root word
all_words = [lemmatizer.lemmatize(word) for word in all_words if word not in ignore_punctuation_list]
all_words = sorted(set(all_words))
all_categories = sorted(set(all_categories))
# Testing to see if lemmatize function is working properly or not
# print(all_words)

# To save those words & categories into pickle files for later on
pickle.dump(all_words, open('all_words.pkl', 'wb'))
pickle.dump(all_categories, open('all_categories.pkl', 'wb'))

# To tern characters, words & categories into numerical numbers, so that we can feed neural network with 1s & 0s
training = []
# Empty output is a template of zeros, as many zeros as there are categories
empty_output_template = [0] * len(all_categories)

# For each of those combinations we're going to create an empty bag of words
# To set the individual words indices to either 0 or 1 depending on if it's occurring in that particular pattern
for all_combinations in all_combinations:
    # To create an empty bag for each combination
    bag = []
    word_patterns = all_combinations[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # To check if the word occurs in the pattern
    for word in all_words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # To copy the list
    output_row = list(empty_output_template)
    # To know the categories with index 1 and set the index in the output_row to 1
    output_row[all_categories.index(all_combinations[1])] = 1
    training.append([bag, output_row])

# To shuffle the training data
random.shuffle(training)
# To tern the training data into numpy array
training = np.array(training)

# To split the numpy array into x and y values to use to train neural network
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# To Create a keras sequential model
training_model = Sequential()
# Add one Dense layer with 128 neurons, input shape dependent on the size of the training data for x, relu activation
training_model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Add one Dropout layer in order to prevent over fitting
training_model.add(Dropout(0.5))
# Add one Dense layer with 64 neurons and relu activation
training_model.add(Dense(64, activation='relu'))
# Add one Dropout layer in order to prevent over fitting
training_model.add(Dropout(0.5))
# Add one Dense layer with neurons as many as there are categories for y, softmax activation to sums up the results in
# the output layer so that they all add up to one, so we get percentages of how likely it is to have that results
training_model.add(Dense(len(train_y[0]), activation='softmax'))
# Applying Adam optimizer to the model, sp it uses momentum and an adaptive learning rate to speed up convergence
adam = adam_v2.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# To Compile the model, with loss function categorical_crossentropy. Also display accuracy metrics
training_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# To feed the same data 800 times into the neural network in a batch size of 6, and verbose of 1 so that we can get
# medium amount of information
history = training_model.fit(np.array(train_x), np.array(train_y), epochs=800, batch_size=6, verbose=1)
training_model.save('training_model.h5', history)
# To display messages when training completed
print("Model training Completed!")
