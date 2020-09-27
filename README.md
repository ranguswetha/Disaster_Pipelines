# This repository has ML pipelines build for categorizing disaster related event messages as part of Disaster Response Pipeline Project for Udacity - Data Science Nanodegree

## Table of Contents

* [Project Motivation](#project-motivation)
* [Installations](#installations)
* [File Descriptions](#file-descriptions)
* [Run](#run)
* [License](#license)



<!--Project Motivation-->
## Project Motivation
This project is part of Udacity's Data Science Nanodegree program. ML pipelines are build to categorize disaster related messages and tweets into caegories where they can be addressed easily.
*	Datasets provided with pre-labelled messages and tweets are processed by builiding ETL pipeline.
*   Data is extracted, transformed and loaded into SQLite DB.
*	Machine learning pipeline is build to train classifier to categorize messages & tweets into provided categories.
*	Flask is used to show the results on WebApp.


<!--Installations-->
## Installations
We need below libraries imported into python 3+ environment to successfully run the code. This repository was written in HTML and python 3+ and requires python and machine learning libraries NumPy, Pandas, os, pickle, nltk, re, json, sqlalchemy.

<!--File descriptions-->
## File descriptions
*   process_data.py: This python script excutes to import CSV dataset provided and performs transformations to the data and then loads the data into SQLite database. 
*   train_classifier.py: This code builds machine learning pipeline to train the classifier to categorize messages available on SQLite database table created in process_data.py and also trains the data provided to build model and saves the same to Pickle file.
*   ETL Pipeline Preparation.ipynb: Development code for process_data.py
*   ML Pipeline Preparation.ipynb: Development code for train_classifier.py
*   run.py: This contains HTML code to utilize pickle file and using Flask app results are showed in realtime.

<!--Run-->
## Run
*   Run the following commands in the project's root directory to set up your database and model.
*   To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
*   To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py
*   Go to http://0.0.0.0:3001/ Or Go to http://localhost:3001/

<!-- License-->
## License
*   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Acknowledgements-->
## Acknowledgements
*   [Udacity](https://www.udacity.com/) 
*   [Figure Eight](https://www.figure-eight.com/)