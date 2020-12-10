import graphviz
from sklearn import tree

from decision_tree_functions import *
from helper_functions import *

# np.random.seed(0)
df_train = generate_data(n=300, n_random_outliers=5)
my_tree = decision_tree_algorithm(df_train, ml_task="classification", max_depth=10)
create_plot(df_train, my_tree, title="Training Data")
plt.title("Model without pruning")
plt.show()

# np.random.seed(7)
df_val = generate_data(n=300)
my_tree_pruned = post_pruning(my_tree, df_train, df_val, ml_task="classification")
create_plot(df_val, my_tree_pruned, title="Validation Data")
plt.title("Model after pruning")
plt.show()

# 2. Titanic Data Set (Classification Task)
# 2.1 Data Preparation
df = pd.read_csv("../data/Titanic.csv")
df["label"] = df.Survived
df = df.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
str_convert_float(df)

# handling missing values
median_age = df.Age.median()
mode_embarked = df.Embarked.mode()[0]
df = df.fillna({"Age": median_age, "Embarked": mode_embarked})

# 2.2 Post-Pruning
max_depth = 16
df_train, df_test = train_test_split(df, test_size=0.15)
df_train, df_val = train_test_split(df_train, test_size=0.15)

my_tree = decision_tree_algorithm(df_train, ml_task="classification", max_depth=max_depth)
my_tree_pruned = post_pruning(my_tree, df_train, df_val, ml_task="classification")

predictions = make_predictions(df_test, my_tree)
predictions_correct = predictions == df_test.label
print(f"Accuracy of Tree:         {predictions_correct.mean()}")
pd.crosstab(df_test.label, predictions, rownames=["label"], colnames=["prediction"])

predictions = make_predictions(df_test, my_tree_pruned)
predictions_correct = predictions == df_test.label
print(f"Accuracy of pruned Tree:  {predictions_correct.mean()}")
pd.crosstab(df_test.label, predictions, rownames=["label"], colnames=["prediction"])

skl_tree = tree.DecisionTreeClassifier(max_depth=max_depth)
skl_tree.fit(df_train.drop(["label"], axis=1), df_train.label)
predictions = skl_tree.predict(df_test.drop(["label"], axis=1))
predictions_correct = predictions == df_test.label
print(f"Accuracy of sklearn Tree: {predictions_correct.mean()}")
pd.crosstab(df_test.label, predictions, rownames=["label"], colnames=["prediction"])

# dot_data = tree.export_graphviz(skl_tree, out_file=None,
#                                 feature_names=df_train.drop(["label"], axis=1).columns,
#                                 class_names="Survived",
#                                 filled=False, rounded=False,
#                                 special_characters=False)
dot_data = tree_graph(my_tree_pruned, {1: "Survived", 0: "Not Survived"})
graph = graphviz.Source(dot_data)
graph.view()

# 3. Bike Rental Data Set (Regression Task)
# 3.1 Data Preparation
df = pd.read_csv("../data/Bike.csv", parse_dates=["dteday"])
df = df.drop(["instant", "casual", "registered"], axis=1)
df = df.rename({"dteday": "date"}, axis=1)

date_column = df.date

df["day_of_year"] = date_column.dt.dayofyear
df["day_of_month"] = date_column.dt.day

df["quarter"] = date_column.dt.quarter
df["week"] = date_column.dt.week

df["is_month_end"] = date_column.dt.is_month_end
df["is_month_start"] = date_column.dt.is_month_start
df["is_quarter_end"] = date_column.dt.is_quarter_end
df["is_quarter_start"] = date_column.dt.is_quarter_start
df["is_year_end"] = date_column.dt.is_year_end
df["is_year_start"] = date_column.dt.is_year_start

df = df.set_index("date")

df["label"] = df.cnt
df = df.drop("cnt", axis=1)

df_train = df.iloc[:-122]
df_val = df.iloc[-122:-61]  # Sep and Oct of 2012
df_test = df.iloc[-61:]  # Nov and Dec of 2012

# 3.2 Post-Pruning
my_tree = decision_tree_algorithm(df_train, ml_task="regression", max_depth=10)
my_tree_pruned = post_pruning(my_tree, df_train, df_val, ml_task="regression")

rss_tree = determine_errors(df_test, my_tree, ml_task="regression")
rss_tree_pruned = determine_errors(df_test, my_tree_pruned, ml_task="regression")

skl_tree = tree.DecisionTreeRegressor(max_depth=max_depth)
skl_tree.fit(df_train.drop(["label"], axis=1), df_train.label)
predictions = skl_tree.predict(df_val.drop(["label"], axis=1))
rss_tree_sklearn = ((predictions - df_val.label) ** 2).sum()

print(f"RSS of Tree:         {int(rss_tree):,}")
print(f"RSS of pruned Tree:  {int(rss_tree_pruned):,}")
print(f"RSS of sklearn Tree: {int(rss_tree_sklearn):,}")

df_plot = pd.DataFrame({"actual": df_test.label,
                        "predictions_tree": make_predictions(df_test, my_tree),
                        "predictions_tree_pruned": make_predictions(df_test, my_tree_pruned),
                        "predictions_tree_sklearn": skl_tree.predict(df_test.drop(["label"], axis=1))})

df_plot.plot(figsize=(18, 6), color=["black", "#66c2a5", "#fc8d62", "#00BFFF"], style=["-", "--", "--", "--"])
plt.show()
