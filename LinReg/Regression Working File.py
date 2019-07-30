import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("/Users/alexchandy13/PycharmProjects/TensorEnvironment/LinReg/student-mat.csv", sep=";")
# print(data.head())

data = data[["G1","G2","G3","studytime","absences","failures"]]
# print(data.head())

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# # Comment Area Start
# best = 0
# for _ in range(100):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     lm = linear_model.LinearRegression()
#     lm.fit(x_train,y_train)
#     acc = lm.score(x_test,y_test)
#     print(acc)
#
#     if acc > best:
#         best = acc
#         with open("/Users/alexchandy13/PycharmProjects/TensorEnvironment/LinReg/studentmodel.pickle","wb") as f:
#             pickle.dump(lm, f)
#         with open("/Users/alexchandy13/PycharmProjects/TensorEnvironment/LinReg/xtest.pickle","wb") as g:
#             pickle.dump(x_test, g)
#         with open("/Users/alexchandy13/PycharmProjects/TensorEnvironment/LinReg/ytest.pickle", "wb") as h:
#             pickle.dump(y_test, h)
#
# # Comment Area End

pickle_sm = open("/Users/alexchandy13/PycharmProjects/TensorEnvironment/LinReg/studentmodel.pickle", "rb")
lm = pickle.load(pickle_sm)
pickle_x = open("/Users/alexchandy13/PycharmProjects/TensorEnvironment/LinReg/xtest.pickle", "rb")
x_test = pickle.load(pickle_x)
pickle_y = open("/Users/alexchandy13/PycharmProjects/TensorEnvironment/LinReg/ytest.pickle", "rb")
y_test = pickle.load(pickle_y)
print("Score: ", lm.score(x_test,y_test))
print("y = ", lm.intercept_," + ", lm.coef_,"x")

predictions = lm.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
#print(predictions)

iv = "G3 Actual"
style.use("ggplot")
pyplot.scatter(y_test,predictions)
pyplot.xlabel(iv)
pyplot.ylabel("Predicted")
pyplot.show()
