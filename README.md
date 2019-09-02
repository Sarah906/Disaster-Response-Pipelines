# Disaster-Response-Pipelines
 ETL Pipeline , ML Pipeline ,Flask Web App

## Motivation

In this project, It will provide disaster responses to analyze data from Figure Eight to build a model for an API that classifies disaster messages.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.


## Files Describation:

data : disaster_categories.csv : it contains disater messages catgoreis
     : disaster_messages.csv: it contains the disaster messages
app: contain files to run the flask web app

process_data.py:  it takes all the CSV files, and creates an SQLite database containing a merged and cleaned version of this data.

train_classifier.py: it takes the SQLite database produced by process_data.py as an input.The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.

ETL Pipeline Preparation.ipynb: This Jupyter notebook was used in the development of process_data.py. process_data.py effectively automates this notebook.

ML Pipeline Preparation.ipynb: This Jupyter notebook was used in the development of train_classifier.py.


## Running Instructions :

**Run process_data.py**
python process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DR.db


**Run train_classifier.py**

python train_classifier.py DB.db models/classifier.pkl


## Results

An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
A Flask app was created to show data visualization and classify the message that user enters on the web page.
