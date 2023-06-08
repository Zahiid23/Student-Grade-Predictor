import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style


data = pd.read_csv("student-mat.csv", sep=";")

data = data[
    [
        "G1",
        "G2",
        "G3",
        "studytime",
        "failures",
        "absences",
        "famrel",
        "health",
        "goout",
        "traveltime",
    ]
]

print(data.head())

predict = "G3"  # label to predict G3 which is the final grade

X = np.array(data.drop(labels=[predict], axis=1))  # all the training data
y = np.array(data[predict])  # predicted G3 values
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.1
)
# loop for training and finding the best model
best = 0
for _ in range(100):  # Running loop 30 times
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.1
    )  # The training model and test size whcih will determine our training data size(10% of data are test samples)

    linear = (
        linear_model.LinearRegression()
    )  # The algorithm we are using is linear regression

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)  # accuracy score
    print("Accuracy:  \n", acc)

    # Only saving model if its accuracy is greater than the best accuracy
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# loading pickel file in order to use same data model(comment out the code above)
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")

# Line of best fit plot
a, b = np.polyfit(data[p], data["G3"], 1)
theta = np.polyfit(data[p], data["G3"], 1)
y_line = theta[1] + theta[0] * data[p]
pyplot.plot(data[p], y_line, "r")

pyplot.show()
