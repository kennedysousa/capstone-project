# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
import pickle

import warnings
warnings.filterwarnings("ignore")

def build_and_train():
    # Load the dataset
    in_file = 'data.csv'
    data = pd.read_csv(in_file, sep=';')

    # Preprocess data - removing unecessary columns
    data = data.drop(['nu_analise', 'nu_unidade', 'nu_unidade_ocorrencia', 'ic_21', 'ic_211', 'ic_212', 'ic_284', 'nu_2841_valor'], axis = 1)
    data = data.drop(['ic_51', 'ic_52a', 'ic_52b', 'ic_52c', 'ic_52d', 'ic_52e', 'ic_52f', 'ic_52g'], axis = 1)
    data = data.dropna(axis=0)
    data = data[data.cd_situacao != 1]
    data = data[data.cd_situacao != 2]
    data = data[data.cd_situacao != 9]
    features_raw = data

    # Log-transform the skewed features
    skewed = ['nu_283_total', 'nu_283_res']
    features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
    
    # Total number of records
    n_records = data.nu_origem.count()
    
    # Number of records that became disciplinary process
    granted = data[(data['cd_situacao']==3)].cd_situacao.count()
    
    # Number of records that did not become disciplinary process 
    declined = data[(data['cd_situacao']==4)].cd_situacao.count()
    
    # Percentage of preliminary analysis that became disciplinary process
    granted_percent = (float(granted)/n_records)*100

    # Print the results
    print ("Total number of records: {}".format(n_records))
    print ("Analysis that became disciplinary process {}".format(granted))
    print ("Analysis archived: {}".format(declined))
    print ("Percentage of disciplinary process admitted: {:.2f}%".format(granted_percent))
    outcomes = data['cd_situacao']
    
    # Convert the output to a universally known format.
    outcomes = outcomes.apply(lambda x: 1 if x == 3 else 0)
    data = data.drop('cd_situacao', axis = 1)
    
    # Split the 'features' and 'income' data into training and testing sets
    print ("Spliting data into training and testing sets")
    X_train, X_test, y_train, y_test = train_test_split(data, outcomes, test_size = 0.2, random_state = 0)
    
    # Show the results of the split
    print ("Training set has {} samples.".format(X_train.shape[0]))
    print ("Testing set has {} samples.".format(X_test.shape[0]))
    
    # Initialize the classifier
    clf = LogisticRegression(random_state=0)
    
    # Create the parameters list you wish to tune
    parameters = {
        'C': [0.001, 0.01, 0.1, 1]
    }
    
    # Make an fbeta_score scoring object
    scorer = make_scorer(fbeta_score, beta=0.5, average='weighted')
    
    # Perform grid search on the classifier using 'scorer' as the scoring method
    grid_obj = GridSearchCV(clf, parameters,scoring=scorer)
    
    # Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(X_train, y_train)
    
    # Get the estimator
    best_clf = grid_fit.best_estimator_
    print (best_clf)
    
    # Make predictions
    best_predictions = best_clf.predict(X_test)
    
    # Report the before-and-afterscores
    print ("\nPerformance Report\n------")
    print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print ("Final F-score on the testing data: {:.4f}\n".format(fbeta_score(y_test, best_predictions, beta = 0.5, average='weighted')))
    print (best_clf)
    
    return best_clf

def create_model():
    model = build_and_train()
    filename = 'model.pk'
    with open('../flask_api/models/'+filename, 'wb') as file:
        pickle.dump(model, file)
    
if __name__ == '__main__':
	model = build_and_train()

	filename = 'model.pk'
	with open('../flask_api/models/'+filename, 'wb') as file:
		pickle.dump(model, file)