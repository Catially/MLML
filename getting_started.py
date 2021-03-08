from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
digits = datasets.load_digits()
print("This is data:")
print(digits.data)
print(type(digits.data))
print(len(digits.data))
print(len(digits.data[0]))
print("This is target:")
print(digits.target)
print(type(digits.target))
print(len(digits.target))

clf = svm.SVC(gamma = 0.001, C = 100.)
clf.fit(digits.data[:-1], digits.target[:-1])
print(clf.predict(digits.data[-1:]))