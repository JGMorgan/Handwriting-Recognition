import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
digits = datasets.load_digits()

for digit in digits.target:
    print(digit)

