import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from naive_bayes_functions import train_test_split, naive_bayes_param, predict, calculate_accuracy, str_convert_float

df = pd.read_csv('data/Iris.csv', index_col=0)
train_data, test_data = train_test_split(df, 0.2)
label_column = test_data.columns[-1]
test_labels = test_data[label_column]
test_data = test_data.drop(label_column, axis=1)

model = naive_bayes_param(train_data)
predict_labels = predict(model, test_data)
print(f'Accuracy of My Naive Bayes: {calculate_accuracy(predict_labels, test_labels)}')
pd.crosstab(test_labels, predict_labels, rownames=[label_column], colnames=["prediction"])

gnb = GaussianNB()
gnb.fit(train_data.drop(label_column, axis=1), train_data[label_column])
predict_labels = gnb.predict(test_data)
print(f'Accuracy of Sklearn Naive Bayes: {calculate_accuracy(predict_labels, test_labels)}')
pd.crosstab(test_labels, predict_labels, rownames=[label_column], colnames=["prediction"])

# Data Preparation
df = pd.read_csv('data/Titanic.csv')
df_labels = df.Survived
label_column = 'Survived'
df = df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
df[label_column] = df_labels
# Handling missing values
median_age = df.Age.median()
mode_embarked = df.Embarked.mode()[0]
df = df.fillna({'Age': median_age, 'Embarked': mode_embarked})

train_data, test_data = train_test_split(df, 0.1)
test_labels = test_data[label_column]
test_data = test_data.drop(label_column, axis=1)

model = naive_bayes_param(train_data)
predict_labels = predict(model, test_data)
print(f'Accuracy of My Naive Bayes: {calculate_accuracy(predict_labels, test_labels)}')
pd.crosstab(test_labels, predict_labels, rownames=[label_column], colnames=["prediction"])

str_convert_float(train_data)
str_convert_float(test_data)
mnb = MultinomialNB()
mnb.fit(train_data.drop(label_column, axis=1), train_data[label_column])
predict_labels = mnb.predict(test_data)
print(f'Accuracy of Sklearn Naive Bayes: {calculate_accuracy(predict_labels, test_labels)}')
pd.crosstab(test_labels, predict_labels, rownames=[label_column], colnames=["prediction"])
