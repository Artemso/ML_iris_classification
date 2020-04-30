import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class	SKlearn():
	def	__init__(self):
		pass
	
	def	k_neighbors_iris(self, data_file):
		data_set = pd.read_csv(data_file, sep=',', header=None)
		data_set.columns = ['sep_len', 'sep_wid', 'pet_len', 'pet_wid', 'class'] # read text file into a dataset and added column names
		x_data = data_set.iloc[:, :-1].values
		y_data = data_set.iloc[:, 4].values # selected x dataset to learn features and y dataset to learn labels
		x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2) # split them into train and test datasets
		knn = KNeighborsClassifier(n_neighbors=1)
		knn.fit(x_train, y_train) # fit the model
		y_pred = knn.predict(x_test) # predict
		print(np.mean(y_test == y_pred))

data = SKlearn()
data.k_neighbors_iris('./data/iris.data')