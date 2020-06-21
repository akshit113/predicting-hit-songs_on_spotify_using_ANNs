__author__ = 'Akshit Agarwal'
__email__ = 'akshit@email.arizona.edu'
__date__ = '2020-06-21'
__dataset__ = 'https://www.kaggle.com/theoverman/the-spotify-hit-predictor-dataset'
__connect__ = 'https://www.linkedin.com/in/akshit-agarwal93/'

from pprint import pprint

import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from pandas import read_csv, get_dummies, concat, DataFrame
from pandas import set_option
from sklearn.metrics import f1_score, jaccard_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def set_pandas():
    # Setting display option in pandas
    set_option('display.max_rows', None)
    set_option('display.max_columns', None)
    set_option('display.width', None)
    set_option('display.max_colwidth', -1)


def check_unique_value(df, colnames):
    """Gets unique value counts for all selected columns in the dataframe including NaN values and PrettyPrints the
    dicitonary
    :param df:
    :param colnames:
    :return: a dictionary
    """
    mydict = {}
    for col in colnames:
        val_count = (df[col].value_counts(dropna=False)).to_dict()
        mydict[col] = val_count
    pprint(mydict)
    return


def one_hot_encode(df, colnames):
    """This function performs one-hot encoding of the columns
    :param df: input df
    :param colnames: columns to be one-hot encoded
    :return: dataframe
    """
    for col in colnames:
        oh_df = get_dummies(df[col], prefix=col)
        df = concat([oh_df, df], axis=1)
        df = df.drop([col], axis=1)
    return df


def fill_missing_values_with_mode(df, colnames):
    """Fills values in the dataframe of list of columns by the mode value
    :param df: Input df
    :param colnames: list of columns
    :return: dataframe
    """
    for col in colnames:
        df[col] = df[col].fillna(df[col].dropna().mode().values[0])
    return df


def fill_missing_values_with_median(df, colnames):
    """Fills values in the dataframe of list of columns by the median value
    :param df: Input df
    :param colnames: list of columns
    :return: dataframe
    """
    for col in colnames:
        df[col] = df[col].fillna(df[col].dropna().median())
    return df


def has_missing_values(df):
    cnt = len(df.columns[df.isnull().any()])
    missing_bool = True if cnt > 0 else False
    if missing_bool:
        print(f'''ALERT! The dataset has missing values..''')
    else:
        print(f'''YAYYY! There no missing values in the dataset.''')
    return


def normalize_columns(df, colnames, scaler):
    for col in colnames:
        # Create x, where x the 'scores' column's values as floats
        x = df[[col]].values.astype(float)
        # Create a minimum and maximum processor object
        # Create an object to transform the data to fit minmax processor
        x_scaled = scaler.fit_transform(x)
        # Run the normalizer on the dataframe
        df[col] = DataFrame(x_scaled)
    print(f'''Normalized Columns: {colnames} using MinMaxScaler.''')
    return


def import_and_clean_data(filename, mode='train'):
    """This function imports the data from a given filename, cleans the dataset and returns a one-hot encoded dataset
    :param filename: filepath of csv
    :param mode: train/test, this paramter determines the shape of one-hot encoded output
    :return: one-hot encoded output
    """
    set_pandas()
    df = read_csv(filename, delimiter=",")
    print(f'''Input Shape: {df.shape}''')
    print(f'''Input head''')
    has_missing_values(df)
    df = df.drop(['track', 'artist', 'uri'], axis=1)
    df['loudness'] = abs(df['loudness'])
    df['loudness'] = df['loudness'] - (df['loudness'].median())
    df['minutes'] = df['duration_ms'] / 60000
    scaler = MinMaxScaler()
    normalize_columns(df, ['tempo', 'chorus_hit', 'sections', 'duration_ms'], scaler)
    # df = fill_missing_values_with_mode(df, ['Credit_History', 'Gender', 'Married', 'Self_Employed', 'Dependents'])
    # df = fill_missing_values_with_median(df, ['Loan_Amount_Term', 'LoanAmount'])
    # df['Dependents'] = to_numeric(df['Dependents'].astype(str).str.replace('+', ''))
    # df = fill_missing_values_with_mode(df, ['Dependents'])
    # df['hasDependents'] = df['Dependents'].apply(lambda x: 0 if x == 0 else 1)
    # df['totalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    # df = df.drop(['Loan_ID'], axis=1)
    # df['Loan_Amount_Term'] = df['Loan_Amount_Term'] / 360
    if mode == 'train':
        df = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                 'duration_ms', 'time_signature', 'chorus_hit', 'sections', 'minutes', 'target']]
    df = df.to_numpy()
    return df


def split_dataset(df, test_size, seed):
    """This function randomly splits (using seed) train data into training set and validation set. The test size
    paramter specifies the ratio of input that must be allocated to the test set
    :param df: one-hot encoded dataset
    :param test_size: ratio of test-train data
    :param seed: random split
    :return: training and validation data
    """
    ncols = np.size(df, 1)
    X = df[:, range(0, ncols - 1)]
    Y = df[:, ncols - 1]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    y_train = get_dummies(y_train)  # One-hot encoding
    y_test = get_dummies(y_test)
    return x_train, x_test, y_train, y_test


def get_model(input_size, output_size, magic='tanh'):
    """This function creates a baseline feedforward neural network with of given input size and output size
        using magic activation function.
    :param input_size: number of columns in x_train
    :param output_size: no of columns in one hpt
    :param magic: activation function
    :return:Sequential model
    """
    mlmodel = Sequential()
    mlmodel.add(Dense(18, input_dim=input_size, activation=magic))
    # kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=l2(1e-4),
    # activity_regularizer=l1(1e-5)))
    # mlmodel.add(LeakyReLU(alpha=0.1))
    mlmodel.add(Dense(64, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(128, activation=magic))
    mlmodel.add(Dense(254, activation=magic))
    mlmodel.add(Dense(324, activation=magic))
    mlmodel.add(Dense(512, activation=magic))

    mlmodel.add(Dense(output_size, activation='softmax'))

    # Setting optimizer
    # mlmodel.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
    opt = SGD(lr=0.001)
    mlmodel.compile(loss="binary_crossentropy", optimizer=opt, metrics=['binary_accuracy'])
    return mlmodel


def fit_and_evaluate(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    """fits the model created in the get_model function on x_train, y_train and evaluates the model performance on
    x_test and y_test using the batch size and epochs paramters

    :param model: Sequential model
    :param x_train: training data
    :param y_train: training label
    :param x_test: testing data
    :param y_test: testing label
    :param batch_size: amount of training data (x_train) fed to the model
    :param epochs: number of times the entire dataset is passed through the network
    :return: tuple of validation_accuracy and validation_loss
    """
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print('Test accuracy:', test_acc)
    print('Test Loss:', test_loss)
    return test_acc, test_loss


def make_predictions(model, x_test, y_test):
    """This function makes predictions using the model on the unseen test dataset
    :param y_test: test labels
    :param model: Sequential model
    :param x_test: unseen test dataset
    :return: predictions in the binary numpy array format
    """

    y_test = y_test.to_numpy()
    y = np.argmax(y_test, axis=-1)
    predictions = model.predict(x_test)
    y_hat = np.argmax(predictions, axis=-1)
    return y_hat, y


def calc_accuracy_using_metrics(y, y_hat, metric, average):
    """This function evaluates the model predictions, y_hat with ground truth y using a sklearn metric
    :param y: ground truth
    :param y_hat: model predictions
    :param metric: evaluation metric (f1_score, precision_score, jaccard_score
    :param average: micro, macro, binary, weighted
    :return: evaluation score
    """
    score = 0
    metrics_list = ['f1_score', 'jaccard_score', 'precision_score']
    average_list = ['micro', 'macro', 'binary']
    if metric not in metrics_list:
        print(f'''{metric} is not a valid metric type. Please try one of these: {metrics_list}''')
        return
    if average not in average_list:
        print(f'''{average} is not a valid average type. Please try one of these: {average_list}''')
        return
    if metric == 'f1_score':
        score = f1_score(y, y_hat, average=average)
    if metric == 'jaccard_score':
        score = jaccard_score(y, y_hat, average=average)
    if metric == 'precision_score':
        score = precision_score(y, y_hat, average=average)
    score = round(score, 4)
    print(f'''{metric}: {score}''')
    return score


def main():
    import warnings
    warnings.filterwarnings("ignore")
    np.random.seed(3)
    warnings.filterwarnings("ignore")
    train_filename = r'dataset-of-10s.csv'
    train_filename = r'combined.csv'

    # test_filename = r'C:\Users\akshitagarwal\Desktop\Keras\datasets\loan\test.csv'
    df_train = import_and_clean_data(train_filename, mode='train')
    # test_df = import_and_clean_data(test_filename, mode='test')
    x_train, x_test, y_train, y_test = split_dataset(df_train, 0.2, 5)
    model = get_model(input_size=16, output_size=2, magic='selu')
    test_acc, test_loss = fit_and_evaluate(model, x_train, y_train, x_test, y_test, 700, 400)
    y_hat, y = make_predictions(model, x_test, y_test)
    # Accuracy can't be calculated, since test labels are no available
    score = calc_accuracy_using_metrics(y, y_hat, 'jaccard_score', 'binary')
    score = calc_accuracy_using_metrics(y, y_hat, 'f1_score', 'binary')
    score = calc_accuracy_using_metrics(y, y_hat, 'precision_score', 'binary')
    print('Program execution complete!')


if __name__ == '__main__':
    main()
