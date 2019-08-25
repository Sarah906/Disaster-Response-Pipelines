import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag ,ne_chunk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_data(database_filepath):
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)
    
    
    # create X and Y dataset
    X = df.message
    Y = df.iloc[:,4:]
    
    # create list containing all category names
    category_names = list(Y.columns.values
    return X, Y, category_names

def tokenize(text):
'''
toknizing text messges
'''
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ",text.lower())
    # tokenize text
    words = word_tokenize(text)
    # remove stop words
    words = [w for w in words if w not in stopwords.words('English')]
    # reduce words to their stems
        stemmed = [PorterStemmer().stem(w) for w in words]
                          
    return stemmed


def build_model():
                          
    # Create pipeline
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer = tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', RandomForestClassifier())
                                ])
                          
            # create parameters dictionary
        parameters = {'clf__max_depth': [1,None],
                          'clf__min_samples_leaf': [1,2],
                          'clf__min_samples_split': [2,5],
                          'clf__n_estimators': [20,40]}
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=1, n_jobs=-1)
                          
        return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
                          
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    for  idx, cat in enumerate(Y_test.columns.values):
    print("{} -- {}".format(cat, accuracy_score(Y_test.values[:,idx], y_pred[:, idx])))
    print("accuracy = {}".format(accuracy_score(Y_test, y_pred)))


def save_model(model, model_filepath):
                          
    with open(model_filepath, 'wb') as file:
    pickle.dump(model, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
