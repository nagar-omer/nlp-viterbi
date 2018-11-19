from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression


class LogLinear:
    def __init__(self, train_src, test_src):
        examples_train, labels_train = load_svmlight_file(train_src)
        examples_test, labels_test = load_svmlight_file(test_src)
        print("loaded")
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')\
            .fit(examples_train, labels_train)
        print("train:" + str(clf.score(examples_train, labels_train)))
        print("test:" + str(clf.score(examples_test, labels_test)))
        e = 0


if __name__ == "__main__":
    LogLinear("ftr_conv_train", "ftr_conv_test")
