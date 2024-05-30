""" !!! TO RUN MODEL, PLACE CCD.xls IN SAME DIRECTORY AS CODE !!! """

from __future__ import print_function
import numpy as np
import seaborn as sns
from keras import Model
from keras.layers import Dense, Dropout, Input, concatenate
from keras.optimizers.legacy import RMSprop
from keras import metrics
import matplotlib.pyplot as plt
import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

np.random.seed(1671)


def load_data():
    """
    Loads the raw credit default data from an .xls file in the same directory as this programme

    @returns: The raw credit default data, separated into features and targets dataframes
    """

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CCD.xls")
    CCD = pd.read_excel(file_path, header=1)

    # separate features from targets
    all_ys = CCD["default payment next month"]
    all_xs = CCD.drop(columns="default payment next month")
    return all_xs, all_ys


def preprocess(xs, ys):
    """
    Applies all relevant preprocessing to the credit default data before training the ML model.
    One-Hot-Encodes all categorical data, and normalizes all continuous data.

    @param xs: The dataframe of all features from the credit defaults dataset
    @param ys: The dataframe of all targets from the credit defaults dataset

    @returns: A binary tuple of the preprocessed xs and ys
    """

    # One-Hot-Encode categorical features
    xs = pd.get_dummies(
        xs,
        columns=[
            "PAY_0",
            "PAY_2",
            "PAY_3",
            "PAY_4",
            "PAY_5",
            "PAY_6",
            "EDUCATION",
            "MARRIAGE",
        ],
    )
    xs["SEX"] -= 1
    # manually add OHE column so all PAY vectors are of equal dimension
    if "PAY_5_1" not in xs.columns:
        xs["PAY_5_1"] = 0
    if "PAY_6_1" not in xs.columns:
        xs["PAY_6_1"] = 0
    # Remove one redundant dimension from each One-Hot-Encoding
    xs = xs.drop(
        columns=[
            col + "_0"
            for col in [
                "EDUCATION",
                "MARRIAGE",
                "PAY_0",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
            ]
        ]
    )

    # remove negative values (assumed erroneous) from bills
    for col in (
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ):
        xs[col] = xs[col].clip(lower=0)

    # apply log base 10 to data following a logarithmic distribution
    # zero values remain as zero
    for col in (
        "LIMIT_BAL",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ):
        xs[col] = np.log10(xs[col].replace(0, np.nan)).replace(np.nan, 0)

    # normalize continuous (non-categorical) features
    for col in (
        "LIMIT_BAL",
        "AGE",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ):
        mean = xs[col].mean(axis=0)
        std = xs[col].std(axis=0)
        xs[col] -= mean
        xs[col] /= std

    # convert all data to type float
    xs = xs.astype("float64")
    ys = ys.astype("float64")
    return xs, ys


def under_sample(xs, ys):
    """
    Applies random undersampling such that the majority and minority class become the same size

    @param xs: The dataframe of all raw features from the credit defaults dataset
    @param ys: The dataframe of all raw targets from the credit defaults dataset

    @returns: A binary tuple of the balanced xs and ys
    """

    under = RandomUnderSampler(sampling_strategy="majority")
    return under.fit_resample(xs, ys)


def split_xs(xs):
    """
    Splits a dataframe of features into two - one with personal features and one with transaction features.

    @param xs: The dataframe of all preprocessed features from the credit defaults dataset

    @returns: A binary tuple of two dataframes - one containing personal features and the other transaction features.
    """

    # Personal data are balance limit, sex, age, education, and marriage
    pers_cols = (
        ["LIMIT_BAL", "SEX", "AGE"]
        + ["EDUCATION_" + str(n) for n in range(1, 7)]
        + ["MARRIAGE_" + str(n) for n in range(1, 4)]
    )
    bal_cols = (xs.drop(columns=pers_cols)).columns
    return xs[pers_cols], xs[bal_cols]


def build_model(pers_len, bal_len):
    """
    Builds a non-sequential model with two branches for processing personal and transaction features independently,
    concatenated and outputted through another set of dense layers.

    @param pers_len: The number of personal data features per sample
    @param bal_len: The number of transaction data features per sample
    @returns: The un-compiled model
    """

    DROPOUT = 0.1
    N_HIDDEN = 124

    pers_input = Input(shape=(pers_len,))
    bal_input = Input(shape=(bal_len,))

    # personal features input branch
    x = Dense(N_HIDDEN * 2, activation="relu")(pers_input)
    x = Dropout(DROPOUT)(x)
    x = Dense(N_HIDDEN, activation="relu")(x)
    x = Dropout(DROPOUT)(x)
    x = Dense(N_HIDDEN, activation="relu")(x)
    x = Dropout(DROPOUT)(x)
    x = Model(inputs=pers_input, outputs=x)

    # transaction features input branch
    y = Dense(N_HIDDEN * 2, activation="relu")(bal_input)
    y = Dropout(DROPOUT)(y)
    y = Dense(N_HIDDEN / 2, activation="relu")(y)
    y = Dropout(DROPOUT)(y)
    y = Model(inputs=bal_input, outputs=y)

    combined = concatenate([x.output, y.output])

    # combined input from personal and transaction branch
    z = Dense(N_HIDDEN, activation="relu")(combined)
    z = Dropout(DROPOUT)(z)
    z = Dense(1, activation="sigmoid")(z)

    return Model(inputs=[x.input, y.input], outputs=z)


def train_model(model, xs, ys):
    """
    Trains a model using k-fold evaluation

    @param model: The model to be trained
    @param xs: The feature dataframe providing the model input
    @param ys: The targets dataframe
    @returns: A tuple of lists containing the loss and accuracy histories
    """

    def get_k_split(data, num_val_samples, k):
        """
        Splits a dataframe for the k-th evaluation

        @param data: The dataframe to be split
        @param num_val_samples: The number of samples per fold
        @param k: The fold number
        @returns: A tuple of the data divided into a testing and a training portion specific to the kth fold
        """

        test_data = data[k * num_val_samples : (k + 1) * num_val_samples - 1]
        # remove testing fold from data
        train_data = data.drop(
            index=range(k * num_val_samples, (k + 1) * num_val_samples)
        )
        return test_data, train_data

    NB_EPOCH = 15
    BATCH_SIZE = 150
    VERBOSE = 1

    # k-fold evaluation
    k = 4
    num_val_samples = len(xs) // k

    history_loss = [[], []]
    history_accuracy = [[], []]

    # get personal and transaction data split
    xs_pers, xs_bal = split_xs(xs)

    for i in range(k):
        print("processing fold #", i)
        x_pers_test, x_pers_train = get_k_split(xs_pers, num_val_samples, i)
        x_bal_test, x_bal_train = get_k_split(xs_bal, num_val_samples, i)
        test_targets, train_targets = get_k_split(ys, num_val_samples, i)

        history = model.fit(
            [x_pers_train, x_bal_train],
            train_targets,
            validation_data=([x_pers_test, x_bal_test], test_targets),
            epochs=NB_EPOCH,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
        )

        history_loss[0] = history_loss[0] + history.history["loss"]
        history_loss[1] = history_loss[1] + history.history["val_loss"]
        history_accuracy[0] = history_accuracy[0] + history.history["accuracy"]
        history_accuracy[1] = history_accuracy[1] + history.history["val_accuracy"]

    return history_loss, history_accuracy


def evaluate_model(model, xs, ys):
    """
    Runs a trained model on testing data, and prints the loss, accuracy, F1 score, precision, and accuracy of the test

    @param model: The trained model to be evaluated
    @param xs: The test dataframe of sample features
    @param ys: The test dataframe of sample targets
    """

    xs_pers, xs_bal = split_xs(xs)
    score = model.evaluate([xs_pers, xs_bal], ys)

    print("Test score:", score[0])
    print("Test accuracy:", score[1])
    print("Test F1 Score:", score[2])
    print("Test precision:", score[3])
    print("Test recall:", score[4])


def plot_histories(histories):
    """
    Plots a graph of different metrics for the training and testing runs

    @param histories: A list of lists. Each sub list contains a binary tuple of the training and testing histories, followed by a string name of the metric recorded in the history
    """

    for history in histories:
        plt.plot(history[0][0]) # training history
        plt.plot(history[0][1]) # testing history
        plt.title("Model " + history[1] + " per Epoch - dropout 0.1")
        plt.ylabel(history[1])
        plt.xlabel("Epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()


def plot_confusion(model, xs, ys):
    """
    Plots a 2x2 confusion matrix from a trained models predicted output compared to the actual targets.

    @param model: The trained model to be tested
    @param xs: The test dataframe of sample features
    @param ys: The test dataframe of sample targets
    """

    xs_pers, xs_bal = split_xs(xs)

    # Round predictions to nearest int
    y_pred_lreg = np.rint(model.predict([xs_pers, xs_bal]))

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(ys, y_pred_lreg)

    # Plot confusion matrix as a heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Reds")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


all_xs, all_ys = load_data()
sampled_xs, sampled_ys = under_sample(all_xs, all_ys)
sampled_xs, sampled_ys = preprocess(sampled_xs, sampled_ys)

# save a small portion of data for getting evaluation metrics
xs_train, x_test_final, ys_train, y_test_final = train_test_split(
    sampled_xs, sampled_ys, test_size=0.1, random_state=42
)

# reset dataframe indices
xs_train = (xs_train.reset_index()).drop(columns="index")
x_test_final = (x_test_final.reset_index()).drop(columns="index")
ys_train = (ys_train.reset_index()).drop(columns="index")
y_test_final = (y_test_final.reset_index()).drop(columns="index")

OPTIMIZER = RMSprop(learning_rate=0.00001)
model = build_model(
    len(split_xs(x_test_final)[0].columns), len(split_xs(x_test_final)[1].columns)
)

model.compile(
    loss="binary_crossentropy",
    optimizer=OPTIMIZER,
    metrics=[
        "accuracy",
        metrics.F1Score(threshold=0.5),
        metrics.Precision(),
        metrics.Recall(),
    ],
)

history_loss, history_accuracy = train_model(model, xs_train, ys_train)
evaluate_model(model, x_test_final, y_test_final)
plot_histories([[history_loss, "Loss"], [history_accuracy, "Accuracy"]])
plot_confusion(model, x_test_final, y_test_final)