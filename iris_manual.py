import pandas as pd
import numpy as np

class	ManualKNN():
	def	__init__(self):
		pass

	def	open_dataset(self, data_file, sep):
		data_set = pd.read_csv(data_file, sep=sep, header=None)
		return(data_set)

	def	split_iris_dataset(self, data_set, train_size=0.8):
		data_set.sample(frac=1)
		data_set = data_set.reindex(np.random.permutation(data_set.index))
		split = int(len(data_set) * train_size)
		train, test = data_set[:split], data_set[split:]
		train = train.to_numpy()
		test = test.to_numpy()
		return train, test

	def	get_euclidean_dist(self, p, q):
		dist = 0
		for x in range(len(p) - 1):
			dist += (p[x] - q[x]) ** 2
		dist = np.sqrt(dist)
		return(dist)

	def	get_k_neighbors(self, row, train, k_neighbors):
		train_dist = []
		for x in train:
			train_dist.append((x, self.get_euclidean_dist(row, x)))
		train_dist.sort(key=lambda pos: pos[1])
		neighbors = []
		for x in range(k_neighbors):
			neighbors.append(train_dist[x][0])
		return(neighbors)

	def	predict(self, row, x_train, k_neighbors):
		neighbors = self.get_k_neighbors(row, x_train, k_neighbors)
		y_train = []
		for x in neighbors:
			y_train.append(x[-1])
		prediction = max(set(y_train))
		return(prediction)

	def	k_neighbors(self, train, test, k_neighbors):
		prediction = []
		for x in test:
			prediction.append(self.predict(x, train, k_neighbors))
		return(prediction)

	def	compare_results(self, prediction, test):
		y_test = []
		for x in test:
			y_test.append(x[-1])
		correct = 0
		for i in range(len(y_test)):
			if y_test[i] == prediction[i]:
				correct += 1
		return correct / (len(y_test)) * 100.0

mkn = ManualKNN()
data_set = mkn.open_dataset('./data/iris.data', ',')
train, test = mkn.split_iris_dataset(data_set, 0.8)
prediction = mkn.k_neighbors(train, test, 5)
print(mkn.compare_results(prediction, test))