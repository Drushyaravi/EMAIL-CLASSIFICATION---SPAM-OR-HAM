# EMAIL-CLASSIFICATION---SPAM-OR-HAM
Spam Text Message Classification
- Overview
This repository contains a Python implementation of a spam text message classification system using machine learning techniques. 
The system uses a dataset of labeled text messages to train and evaluate two classification models: Naive Bayes and Decision Tree.

- Dataset
The dataset used in this project is the "SPAM text message 20170820 - Data.csv" file, which contains 5572 labeled text messages (ham or spam).

- Preprocessing
The text data is preprocessed using the following steps:

Remove non-alphabetic characters
Convert to lowercase
Tokenize the text
Remove stop words
Stemming using Porter Stemmer
Feature Extraction
The preprocessed text data is then converted into TF-IDF vectors using the TfidfVectorizer from scikit-learn.

- Feature Selection
The TF-IDF vectors are then selected using the Best First Feature Selection algorithm with k=500 features.

- Classification Models
Two classification models are implemented:

- Naive Bayes (MultinomialNB)
Decision Tree (DecisionTreeClassifier with entropy criterion)
Evaluation
The performance of each model is evaluated using accuracy score and confusion matrix.

- Results
The results of the evaluation are:

Naive Bayes Accuracy: 0.9444
J48 (Decision Tree) Accuracy: 0.9641
- Visualization
The confusion matrices for each model are visualized using seaborn and matplotlib.

- Requirements
Python 3.x
scikit-learn
nltk
pandas
seaborn
matplotlib
Usage
Clone the repository
Install the required packages
Run the spam_text_message_classification.py file


Author
M R DRUSHYA

Acknowledgments
The dataset is from Kaggle: https://www.kaggle.com/uciml/sms-spam-collection-dataset
The implementation is inspired by various online resources and tutorials.
