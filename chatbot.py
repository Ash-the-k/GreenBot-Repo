import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import re
from nltk.stem import WordNetLemmatizer

from pathlib import Path  
repo_root = Path(__file__).parent

# Train Intent Model
def trainIntentModel():
    # Load the 'intent.csv' dataset and prepare it to train the model
    dataset = pd.read_csv(repo_root / 'datasets' / 'intent.csv', names=["Query", "Intent"])
    X = dataset["Query"]  # Extract the queries
    y = dataset["Intent"]  # Extract the intents

    unique_intent_list = list(set(y))  # List of unique intents
    print("Intent Dataset successfully loaded!")

    # Clean and prepare the intents corpus
    queryCorpus = []
    lemmatizer = WordNetLemmatizer()

    for query in X:
        query = re.sub('[^a-zA-Z]', ' ', query)  # Remove non-alphabetic characters
        query = query.split()  # Split the query into words
        tokenized_query = [lemmatizer.lemmatize(word.lower()) for word in query]  # Lemmatize each word
        tokenized_query = ' '.join(tokenized_query)  # Join words back into a string
        queryCorpus.append(tokenized_query)  # Add the cleaned query to the corpus

    print(queryCorpus)
    print("Corpus created")

    # Convert text to numerical data using CountVectorizer
    countVectorizer = CountVectorizer(max_features=800)
    corpus = countVectorizer.fit_transform(queryCorpus).toarray()  # Transform the corpus into a matrix of token counts
    print(corpus.shape)
    print("Bag of words created!")

    # Save the CountVectorizer for later use
    (repo_root / 'saved_state').mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    pk.dump(countVectorizer, open(repo_root / 'saved_state' / 'IntentCountVectorizer.sav', 'wb'))
    print("Intent CountVectorizer saved!")
    

    # Encode the intent classes into numeric labels
    labelencoder_intent = LabelEncoder()
    y = labelencoder_intent.fit_transform(y)  # Transform the intent labels to numerical values
    y = to_categorical(y)  # Convert to one-hot encoding
    print("Encoded the intent classes!")
    print(y)

    # Create a dictionary mapping each intent to its numerical label
    intent_label_map = {cl: labelencoder_intent.transform([cl])[0] for cl in labelencoder_intent.classes_}
    print(intent_label_map)
    print("Intent Label mapping obtained!")

    # Initialize the Artificial Neural Network
    classifier = Sequential()
    classifier.add(Dense(units=128, activation='relu', input_shape=(corpus.shape[1],)))  # Input layer
    classifier.add(Dropout(0.5))  # Dropout layer for regularization
    classifier.add(Dense(units=64, activation='relu'))  # Hidden layer
    classifier.add(Dropout(0.5))  # Dropout layer for regularization
    classifier.add(Dense(units=y.shape[1], activation='softmax'))  # Output layer
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model
    classifier.fit(corpus, y, batch_size=128, epochs=700)  # Train the model

    return classifier, intent_label_map

intent_model, intent_label_map = trainIntentModel()
intent_model.save(repo_root / 'saved_state' / 'intent_model.h5')
print("Intent model saved!")

def trainEntityModel():
    # Load the 'data-tags.csv' dataset and prepare it to train the model
    dataset = pd.read_csv(repo_root / 'datasets' / 'data-tags.csv', names=["word", "label"])
    X = dataset.iloc[:, :-1].values  # Extract the words
    y = dataset.iloc[:, 1].values  # Extract the labels
    print("Entity Dataset successfully loaded!")

    entityCorpus = []
    lemmatizer = WordNetLemmatizer()

    for word in X.astype(str):
        word = [lemmatizer.lemmatize(word[0])]  # Lemmatize each word
        entityCorpus.append(word)  # Add the lemmatized word to the corpus

    print(entityCorpus)
    X = np.array(entityCorpus).reshape(len(entityCorpus),)  # Reshape the corpus

    # Convert the text data to a bag of words model
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(X).toarray()  # Transform the corpus into a matrix of token counts
    print("Entity Bag of words created!")

    # Save the CountVectorizer for later use
    pk.dump(cv, open(repo_root / 'saved_state' / 'EntityCountVectorizer.sav', 'wb'))
    print("Entity CountVectorizer state saved!")

    # Encode the entity labels into numeric labels
    labelencoder_y = LabelEncoder()
    y = labelencoder_y.fit_transform(y.astype(str))  # Transform the entity labels to numerical values
    print("Encoded the entity classes!")

    # Create a dictionary mapping each entity to its numerical label
    entity_label_map = {cl: labelencoder_y.transform([cl])[0] for cl in labelencoder_y.classes_}
    print(entity_label_map)
    print("Entity Label mapping obtained!")

    # Train the Naive Bayes classifier
    classifier = GaussianNB()
    classifier.fit(X, y)  # Fit the model
    print("Entity Model trained successfully!")
    
    (repo_root / 'saved_state').mkdir(parents=True, exist_ok=True)
    # Save the trained model
    pk.dump(classifier, open(repo_root / 'saved_state' / 'entity_model.sav', 'wb'))
    print("Trained entity model saved!")

    return entity_label_map

entity_label_map = trainEntityModel()
loadedEntityCV = pk.load(open(repo_root / 'saved_state' / 'EntityCountVectorizer.sav', 'rb'))
loadedEntityClassifier = pk.load(open(repo_root / 'saved_state' / 'entity_model.sav', 'rb'))

def getEntities(query):
    # Transform the query to a bag of words model
    query = loadedEntityCV.transform(query).toarray()
    # Predict the entity tags
    response_tags = loadedEntityClassifier.predict(query)

    # Map the predicted tags to their respective entities
    entity_list = [list(entity_label_map.keys())[list(entity_label_map.values()).index(tag)] for tag in response_tags if tag in entity_label_map.values()]
    return entity_list

import json
import random

# Load the intents JSON file
with open(repo_root / 'datasets' / 'intents.json') as json_data:
    intents = json.load(json_data)

loadedIntentClassifier = load_model(repo_root / 'saved_state' / 'intent_model.h5')
loaded_intent_CV = pk.load(open(repo_root / 'saved_state' / 'IntentCountVectorizer.sav', 'rb'))

def process_query(user_query):
    # Clean and preprocess the user query
    query = re.sub('[^a-zA-Z]', ' ', user_query)
    query = query.split()
    lemmatizer = WordNetLemmatizer()
    tokenized_query = [lemmatizer.lemmatize(word.lower()) for word in query]
    processed_text = ' '.join(tokenized_query)
    processed_text = loaded_intent_CV.transform([processed_text]).toarray()

    # Predict the user's intent
    predicted_Intent = loadedIntentClassifier.predict(processed_text)
    result = np.argmax(predicted_Intent, axis=1)

    # Find the intent label from the predicted result
    USER_INTENT = [key for key, value in intent_label_map.items() if value == result[0]][0]

    # Print a response for the predicted intent
    response = ""
    for i in intents['intents']:
        if i['tag'] == USER_INTENT:
            response = random.choice(i['responses'])

    # Extract entities from the user query
    entities = getEntities(tokenized_query)
    return response, entities

# Example function to simulate response retrieval
def get_response(user_query):
    response, entities = process_query(user_query)
    return response



