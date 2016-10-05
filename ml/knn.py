import numpy as np
import math
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier as baseline

def euclidean_distance(x, y, numpy=True):
	"""
	x: numpy array of first coordinate
	y: numpy array of second coordinate
	"""

	if numpy:
		# alternatively
		# np.sqrt(np.sum(x - y) ** 2)
		return np.linalg.norm(x - y)
	else:
		return sum(((a - b) ** 2 for a, b in zip(x, y)))


def majority(top_k_labels):
	assert isinstance(top_k_labels, list), "Need to pass a list"
	return max(set(top_k_labels), key=top_k_labels.count)


def knn_v01(xs, ys, query, k):
	distances = [(euclidean_distance(x, y), y) for x, y in zip(xs, ys)]
	distances_sorted = sorted(distances, key=lambda x: x[0])
	top_k_labels = [d[1] for d in distances[:k]]

	return majority(top_k_labels)


def baseline_prediction(x, y, query, baseline, k=1, algorithm='brute'):
	baseline_model = baseline(k, algorithm=algorithm)
	baseline_model.fit(x, y)
	return baseline_model.predict([query])


def test_helpers():
	"tests helper functions"

	assert majority([1,1,1]) == 1
	assert majority([1,1,1,2]) == 1
	assert majority([1,1,1,1,1,0]) == 1
	assert majority([1,0,0,0,0]) == 0
	assert majority([1]) == 1
	assert majority([1,1,2,2]) == 1  # ???
	assert majority([10,12]) == 10

	assert euclidean_distance(np.array([1,1]), np.array([1,1])) == 0
	assert euclidean_distance(np.array([1,1]), np.array([0,0])) == math.sqrt(2)
	assert euclidean_distance(np.array([3,5]), np.array([4,4])) == math.sqrt(2)
	assert euclidean_distance(np.array([10,100]), np.array([50,-50])) == math.sqrt(40 ** 2 + 150 ** 2)
	assert euclidean_distance(np.array([0,0]), np.array([-1,1])) == math.sqrt(2)

	print "All unit tests passed"


def test():
	"tests the main algorithm"

	# version to test
	implementations = [knn_v01]
	knn = implementations[-1]

	# dataset
	iris = datasets.load_iris()
	x = iris.data[:, :2]  # for simplicity and viz
	y = iris.target
	
	# test for correctness
	query = np.array([0, 0])
	print knn(x, y, query, 1), baseline_prediction(x, y, query, baseline, 1)
	assert knn(x, y, query, 1) == baseline_prediction(x, y, query, baseline, 1)
	assert knn(x, y, query, 2) == baseline_prediction(x, y, query, baseline, 2)
	assert knn(x, y, query, 5) == baseline_prediction(x, y, query, baseline, 5)

	query = np.array([10, 10])
	print baseline_prediction(x, y, query, baseline, 1), knn(x, y, query, 1)
	assert knn(x, y, query, 1) == baseline_prediction(x, y, query, baseline, 1)
	assert knn(x, y, query, 2) == baseline_prediction(x, y, query, baseline, 2)
	assert knn(x, y, query, 5) == baseline_prediction(x, y, query, baseline, 5)

	# performance test

	print "All integration tests passed"


if __name__ == '__main__':
	test_helpers()
	test()