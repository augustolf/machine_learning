import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()

def export_pdf(clf):
  import graphviz
  dot_data = tree.export_graphviz(clf, out_file=None)
  graph = graphviz.Source(dot_data)
  graph.render("iris")

  dot_data = tree.export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)
  graph = graphviz.Source(dot_data)
  graph.render("iris2")

def trainAClassifierAndTestingData():
  test_idx = [0, 50, 100]

  # training data

  train_target = np.delete(iris.target, test_idx)
  train_data = np.delete(iris.data, test_idx, axis=0)

  clf = tree.DecisionTreeClassifier()
  clf.fit(train_data, train_target)

  # testing data

  test_target = iris.target[test_idx]
  test_data = iris.data[test_idx]

  print test_target
  print test_data

  print clf.predict(test_data)

  export_pdf(clf)


def printIrisDataSet():
  print iris.feature_names
  print iris.target_names
  print iris.data[0]
  print iris.target[0]

  for i in range(len(iris.target)):
    print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

#printIrisDataSet()

trainAClassifierAndTestingData()

