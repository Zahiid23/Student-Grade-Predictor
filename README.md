# Student-Grade-Predictor
Prediction of student grades using Machine Learning

## Description

This machine learning model uses linear regression to draw a best fit line. It predicts studenst Final grades (G3) on based on a number of factors but mainly their previous G1(grade 1) and G2 (grade 2). Other factors such as absence, failures, family support, school support are also used.

![](https://github.com/Zahiid23/Student-Grade-Predictor/tree/main/displays/G1-FinalGrdRel.png) ![](https://github.com/Zahiid23/Student-Grade-Predictor/blob/main/displays/G2-G3Rel.png)

## Files required

student-mat.csv
[can be downlaoded]https://archive.ics.uci.edu/dataset/320/student+performance 


## Getting Started

Clone this repository using the following command: </p>

```
$ git clone https://github.com/Zahiid23/Student-Grade-Predictor.git

```
To add path of CSV file:

```
$ data = pd.read_csv("< PATH >/student-mat.csv", sep=";")

```

## Dependencies

* Python
* Pandas
* numpy
* matplotlib
* Sklearn
* pickle (comes with python by default)

## References
All data used for the project was acquired from the UC Irving Machine Learning Repository
https://archive.ics.uci.edu/

