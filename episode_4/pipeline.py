#import dataset
from sklearn import datasets
iris = datasets.load_iris()

# flower measurements
X = iris.data
# flower labels
y = iris.target

from sklearn.model_selection import train_test_split
# metade base de treino, outra metade de teste (0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier() 

my_classifier.fit(X_train, y_train)
predictions = my_classifier.predict(X_test)



print predictions


from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)