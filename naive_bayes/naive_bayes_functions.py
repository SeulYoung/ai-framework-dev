import random
from math import exp, sqrt, pi

import pandas as pd


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
    test_indices = random.sample(population=df.index.tolist(), k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


def naive_bayes_param(train_df):
    label_column = train_df.columns[-1]
    data_df = train_df.drop(label_column, axis=1)
    continuous_columns = data_df.select_dtypes(include="number").columns
    categorical_columns = data_df.select_dtypes(exclude="number").columns
    # Group by label
    grouped = train_df.groupby(label_column)
    bayes_param = list()

    # Calculate the mean and std of continuous features
    if continuous_columns.size != 0:
        bayes_param.append(grouped[continuous_columns].mean())
        bayes_param.append(grouped[continuous_columns].std())
    else:
        bayes_param.append([])
        bayes_param.append([])

    # Calculate the probability of categorical features
    if categorical_columns.size != 0:
        result = grouped[categorical_columns].apply(lambda x: pd.DataFrame(x.value_counts() / x.size))
        # bayes_param.append(result.reset_index())
        bayes_param.append(result)
    else:
        bayes_param.append([])

    bayes_param.append(grouped.size())
    return bayes_param


# Calculate the Gaussian probability distribution function for x
def calculate_gaussian_probability(x, mean, std):
    exponent = exp(-((x - mean) ** 2 / (2 * std ** 2)))
    return (1 / (sqrt(2 * pi) * std)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(row, summaries, categorical_columns):
    probabilities = summaries[3] / summaries[3].sum()
    for label in summaries[3].index:
        # Calculate categorical columns
        if categorical_columns.size != 0:
            row_value = [label]
            for column in categorical_columns:
                row_value.append(row[column])
            probabilities[label] *= summaries[2].loc[tuple(row_value)]
        # Calculate continuous columns
        for column in summaries[0].columns:
            mean = summaries[0].loc[label, column]
            std = summaries[1].loc[label, column]
            probabilities[label] *= calculate_gaussian_probability(row[column], mean, std)
    return probabilities.idxmax()


# Predict the class for a given data set by row
def predict(summaries, test_df):
    categorical_columns = test_df.select_dtypes(exclude="number")
    return test_df.apply(calculate_class_probabilities, args=(summaries, categorical_columns), axis=1)


def calculate_accuracy(predict_labels, test_labels):
    return predict_labels[predict_labels == test_labels].size / predict_labels.size


def str_convert_float(df):
    columns = df.select_dtypes(exclude="number").columns
    for col_name in columns:
        unique_values = df[col_name].unique()
        for i in range(len(unique_values)):
            df.loc[df[col_name] == unique_values[i], col_name] = i
