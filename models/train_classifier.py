# import libraries
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# import libraries
import numpy as np
import pandas as pd

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    """
    Load Data from SQLite table
    
    Inputs:
        database name
    Output:
        X: messages
        Y:  all other fields
        category names for visualization purpose
    """
    
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    
    # Given value 2 in the related field are neglible so it could be error. Replacing 2 with 1 to consider it a valid response.
    # Alternatively, we could have assumed it to be 0 also. In the absence of information I have gone with majority class.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df['message']
    y = df.iloc[:,4:]
    
    category_names = y.columns 
    return X, y, category_names


def tokenize(text):
    """
    Tokenize and normalize
    
    """
    
    tokens = nltk.word_tokenize(text)
    
    lemmatizer = nltk.WordNetLemmatizer()

    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
      
    Class extracts starting verb if each sentence and utilized while adding additional feature to ML classifier.
    
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Building Pipeline 
    
    Output:
        ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    #Using GridSearchCV as found more accurate in identifying parameters for classifier.
    parameters = {
              'clf__estimator__n_estimators': [10, 20,]
                #,'features__text_pipeline__vect__ngram_range': ((1,1), (1,2))
             }

    cv = GridSearchCV(estimator= pipeline, param_grid= parameters, scoring='f1_micro', n_jobs=-1)

    return cv

def multioutput_fscore(y_true,y_pred,beta=1):
    """
    MultiOutput Fscore
    
    This is used to calculate mean F1 score and accuracy from classification_report, to store the results into dataframe and 
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Inputs:
        y_true: labels
        y_prod:  predictions
        beta: Beta value to be used to calculate fscore metric
    
    Output:
        Calculation geometric mean of fscore
    """
    
    # If provided y predictions is a dataframe then extract the values from that
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    
    # If provided y actuals is a dataframe then extract the values from that
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    
    f1score_list = []
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        f1score_list.append(score)
        
    f1score = np.asarray(f1score_list)
    f1score = f1score[f1score<1]
    
    # Get the geometric mean of f1score
    f1score = gmean(f1score)
    return f1score

def evaluate_model(model, X_test, Y_test, category_names):
    """
   
    Inputs:
        model 
        X_test 
        Y_test 
        category_names 
    Outputs:
        Scores
    """
    Y_pred = model.predict(X_test)
    
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%'.format(multi_f1*100))

    # To Print the whole classification report.
    #Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    #for column in Y_test.columns:
        #print('Model Performance with Category: {}'.format(column))
        #print(classification_report(Y_test[column],Y_pred[column]))


def save_model(model, model_filepath):
    """
       save model to pickle file
    
    Inputs:
        pipeline: GridSearchCV or Scikit Pipelin object
        pickle_filepath:  destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
         Extract data from SQLite db
         Train ML model on training set
         Estimate model performance on test set
         Save trained model as Pickle
    
    """

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