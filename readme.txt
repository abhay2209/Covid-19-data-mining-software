These are two ways to run the code.
1. Python file (.py)  (m3.py)
You will first need to get the data from both files (data1 and data2) from the
provided GitHub link. Our final data was big in size so we split the data into two parts. The code automatically combines it back. You can then run the .py file from your command line.
Github for data: https://github.com/aamritpa/CMPT-Data-Covid-19


2.  Jupyter notebook(.ipynb) (m3.ipynb)
You can run it using the file using the Jupyter notebook provided too. This gets the data
from GitHub so you won’t have to worry about getting the data in your working directory.


Both will ask if you want to check for GridSearch, If yes, it will produce similar .csv files in the results folder, and train for best parameters then print the train, test score, and classification report for each model. Else, It will train for the best parameters and print the train, test score, and classification report for each model.

Note**

Results for each GridSearchCV is in Results folder named as KNN for K-Neighbour, RF- Random Forest and XGB - XGBoost.
Results from the GridSearchCV are combined together in Results/results.pdf 
