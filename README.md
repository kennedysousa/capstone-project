
# Capstone Project
## Machine Learning Engineer Nanodegree
___
### Kennedy Sousa
December 23th, 2017
___


## Project Overview

Bank ACME’s Office of Compliance is a department responsible for monitoring the activities and conduct of employees: whenever an irregularity is detected, the bank must analyze whether the irregularity stems from misconduct or weaknesses in the process, in order to mitigate the operational risk and apply the penalty to those involved, if applicable, including possible compensation for financial losses. 

The procedure starts with a process called preliminary analysis that consists in an investigation and aims to gather information about the issue, like authorship, which rule was broken, description of the facts, value involved, etc. After all the relevant information is gathered, the final report and the chain of evidence are sent to decision-making authority for deliberation. If the case is admitted, the indictee becomes defendant and is subject to penalties like written reprimand, suspension and discharge. 

This project addresses the real world problem of identifying whether the case will be admitted or not, based in some multiple-choice fields filled in the report.

## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

	```	
		git clone https://github.com/kennedysousa/capstone-project.git
		cd capstone-project
	```
2. Obtain the necessary Python packages
	For __Linux__:
	```
		conda env create -f flask_api.yml
		source activate flask_api
	```
3. In the terminal run 
    ```
        gunicorn --bind 0.0.0.0:8000 server:app
    ```
    
4. Open the notebook and follow the instructions
    ```
        jupyter notebook client.ipynb
    ```
    
