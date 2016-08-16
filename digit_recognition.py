import math
import warnings
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
knn = KNeighborsClassifier()
digits = datasets.load_digits()
Success = 0

knn.fit(digits.data[:len(digits.data)/2], digits.target[:len(digits.target)/2])

for i in range(math.ceil(len(digits.data)/2)):
    print("predicted:", knn.predict(digits.data[i + len(digits.data)/2])[0], "actual:", digits.target[i + len(digits.data)/2])
    if ( knn.predict(digits.data[i + len(digits.data)/2])[0] == digits.target[i + len(digits.data)/2] ):
        Success += 1

Accuracy = (Success / (math.ceil(len(digits.data)/2))) * 100
Print(Accuracy, "%")
