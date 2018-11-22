import pickle
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression


class LogLinear:
    def __init__(self, train_src=None, model_pkl=None):
        self._model = pickle.load(open(model_pkl, "rb")) if model_pkl else \
            LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        examples_train, labels_train = load_svmlight_file(train_src) if train_src else (None, None)
        if train_src:
            print("files - loaded \nstart training")
            self._model.fit(examples_train, labels_train)
            print("train:" + str(self._model.score(examples_train, labels_train)))

    def create_model_file(self, out_name="model_file"):
        pickle.dump(self._model, open(out_name, "wb"))

    def score(self, data_src):
        examples, labels = load_svmlight_file(data_src)
        return self._model.score(examples, labels)

    def predict(self, vec):
        e = 0
        return self._model.predict(vec)


if __name__ == "__main__":
    LogLinear(train_src="features_sparse").create_model_file()


