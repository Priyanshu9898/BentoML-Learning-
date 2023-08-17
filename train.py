import bentoml

from sklearn import svm
from sklearn import datasets

# Dataset
iris = datasets.load_iris()

x , y = iris.data, iris.target

# Train Model
clf = svm.SVC(gamma = 'scale')
clf.fit(x , y)


# Save Model
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print("saved model: ", saved_model)


# model: iris_clf:2iazdqr4xwzkgdew