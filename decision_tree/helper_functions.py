import random
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("darkgrid")


def str_convert_float(df):
    columns = df.select_dtypes(exclude="number").columns
    for col_name in columns:
        unique_values = df[col_name].unique()
        for i in range(len(unique_values)):
            df.loc[df[col_name] == unique_values[i], col_name] = i


def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df


def generate_data(n, specific_outliers=None, n_random_outliers=None):
    # create data
    if specific_outliers is None:
        specific_outliers = []
    data = np.random.random(size=(n, 2)) * 10
    data = data.round(decimals=1)
    df = pd.DataFrame(data, columns=["x", "y"])
    df["label"] = df.x <= 5

    # add specific outlier data points
    for outlier_coordinates in specific_outliers:
        df = df.append({"x": outlier_coordinates[0],
                        "y": outlier_coordinates[1],
                        "label": True},
                       ignore_index=True)

    # add random outlier data points
    if n_random_outliers:
        outlier_x_values = (6 - 5) * np.random.random(size=n_random_outliers) + 5  # value between 5 and 6
        outlier_y_values = np.random.random(size=n_random_outliers) * 10

        df_outliers = pd.DataFrame({"x": outlier_x_values.round(decimals=2),
                                    "y": outlier_y_values.round(decimals=2),
                                    "label": [True] * n_random_outliers})
        df = df.append(df_outliers, ignore_index=True)

    return df


def plot_decision_boundaries(tree, x_min, x_max, y_min, y_max):
    color_keys = {True: "orange", False: "blue"}
    # recursive part
    if isinstance(tree, dict):
        question = list(tree.keys())[0]
        yes_answer, no_answer = tree[question]
        feature, _, value = question.split()

        if feature == "x":
            plot_decision_boundaries(yes_answer, x_min, float(value), y_min, y_max)
            plot_decision_boundaries(no_answer, float(value), x_max, y_min, y_max)
        else:
            plot_decision_boundaries(yes_answer, x_min, x_max, y_min, float(value))
            plot_decision_boundaries(no_answer, x_min, x_max, float(value), y_max)
    # "tree" is a leaf
    else:
        plt.fill_between(x=[x_min, x_max], y1=y_min, y2=y_max, alpha=0.2, color=color_keys[tree])


def create_plot(df, tree=None, title=None):
    sns.lmplot(data=df, x="x", y="y", hue="label", fit_reg=False, height=4, aspect=1.5, legend=False)
    plt.title(title)

    if tree or tree is False:  # root of the tree might just be a leave with "False"
        x_min, x_max = round(df.x.min()), round(df.x.max())
        y_min, y_max = round(df.y.min()), round(df.y.max())

        plot_decision_boundaries(tree, x_min, x_max, y_min, y_max)


def tree_graph(tree, class_names):
    out_graph = StringIO()
    out_graph.write('digraph Tree {\n')

    # Specify node aesthetics
    out_graph.write('node [shape=box] ;\n')
    # Specify graph & edge aesthetics
    out_graph.write('edge [fontname=helvetica] ;\n')
    tree_graph_recurse(tree, out_graph, class_names)
    out_graph.write("}")
    return out_graph.getvalue()


def tree_graph_recurse(tree, out_graph, class_names, parent=None):
    question = list(tree.keys())[0]
    node_id = tree['node_id']
    left_child, right_child = tree[question]

    out_graph.write('%d [label="%s"] ;\n' % (node_id, question))
    # if not isinstance(left_child, dict):
    #     out_graph.write('\nclass = %s' % left_child)
    # if not isinstance(right_child, dict):
    #     out_graph.write('\nclass = %s' % right_child)

    if parent is not None:
        # Add edge to parent
        out_graph.write('%d -> %d' % (parent, node_id))
        if parent == 0:
            # Draw True/False labels if parent is root node
            out_graph.write(' [labeldistance=2.5, labelangle=')
            if node_id == 1:
                out_graph.write('45, headlabel="True"]')
            else:
                out_graph.write('45, headlabel="False"]')
        out_graph.write(' ;\n')
    else:
        global max_node_id
        max_node_id = 999

    if isinstance(left_child, dict):
        tree_graph_recurse(left_child, out_graph, class_names, node_id)
    else:
        out_graph.write('%d [label="class = %s"] ;\n' % (max_node_id, class_names[left_child]))
        out_graph.write('%d -> %d ;\n' % (node_id, max_node_id))
        max_node_id += 1

    if isinstance(right_child, dict):
        tree_graph_recurse(right_child, out_graph, class_names, node_id)
    else:
        out_graph.write('%d [label="class = %s"] ;\n' % (max_node_id, class_names[right_child]))
        out_graph.write('%d -> %d ;\n' % (node_id, max_node_id))
        max_node_id += 1
