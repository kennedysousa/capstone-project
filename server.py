#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:48:06 2017

@author: mlnd
"""

import pandas as pd
import pickle
from flask import Flask, jsonify, request
import utils
import os.path

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    """API Call
    
    Pandas dataframe (sent as a payload) from API Call
    """
    try:
        request_json = request.get_json()
        test = pd.read_json(request_json, orient='records')
        analisys_id = test['cd_analise']
        test = test.drop('cd_analise', axis = 1)
    except Exception as e:
        raise e
        
    clf = 'model.pk'
    
    if test.empty:
        return(bad_request())    
    else:
		#Load the saved model
        print("Loading the model...")
        loaded_model = None
        
        if not os.path.exists('./models/'+clf):
            utils.create_model()
        
        with open('./models/'+clf,'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded... \nReady to make predictions...")
        predictions = loaded_model.predict(test)
        
        prediction_series = list(pd.Series(predictions))
        
        final_predictions = pd.DataFrame(list(zip(analisys_id, prediction_series)))
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200
        
        return (responses)

@app.errorhandler(400)
def bad_request(error=None):
	message = {
			'status': 400,
			'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
	}
	resp = jsonify(message)
	resp.status_code = 400

	return resp

@app.errorhandler(500)
def server_error(error=None):
	message = {
			'status': 500,
			'message': 'Server error! Application failed due to an error on server side.',
	}
	resp = jsonify(message)
	resp.status_code = 500

	return resp