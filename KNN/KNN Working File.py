import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("/Users/alexchandy13/PycharmProjects/TensorEnvironment/KNN/car.data")
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train,y_train)
acc = knn.score(x_test,y_test)
print(acc)

names = ["unacc","acc","good","vgood"]
predicted = knn.predict(x_test)

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], ", Data: ", x_test[x], ", Actual: ", names[y_test[x]])
