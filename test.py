import bentoml


iris_clf_runner = bentoml.sklearn.get('iris_clf:latest').to_runner()
iris_clf_runner.init_local()
print(iris_clf_runner.predict.run([[5.9, 3.1, 2.6, 8.7]]))
