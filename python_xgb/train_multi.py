import os
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
LOCAL_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
DATASET_PATH = PARENT_PROJ_PATH + '/dataset'
MODEL_PATH = LOCAL_PROJ_PATH + '/multi_model'


def generate_train_test_data():
    df = pd.read_csv(DATASET_PATH + '/iris.data', header=None)
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    str_label = 'class'
    df.columns = features + [str_label]
    train, test, y_train, y_test = train_test_split(
        df[features], df[str_label], test_size=0.1, random_state=0)
    train[str_label] = y_train
    test[str_label] = y_test
    train.to_csv(DATASET_PATH + '/iris_train.csv', index=0)
    test.to_csv(DATASET_PATH + '/iris_test.csv', index=0)


def main():
    # generate_train_test_data()

    train = pd.read_csv(DATASET_PATH + '/iris_train.csv')
    test = pd.read_csv(DATASET_PATH + '/iris_test.csv')

    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    str_label = 'class'
    label = 'label'

    le = preprocessing.LabelEncoder().fit(train[str_label])
    train[label] = le.transform(train[str_label])
    test[label] = le.transform(test[str_label])

    train.to_csv(DATASET_PATH + '/iris_train.csv', index=0)
    test.to_csv(DATASET_PATH + '/iris_test.csv', index=0)

    d_train = xgb.DMatrix(train[features], train[label], missing=np.nan)
    d_test = xgb.DMatrix(test[features], test[label], missing=np.nan)

    params = {
        "eta": 0.1, "eval_metric": "mlogloss",
        "gamma": 0, "max_depth": 5, "min_child_weight": 1.0,
        "objective": "multi:softprob", "seed": 0,
        "num_class": 3
    }

    plst = params.items()  # turn to tuple
    evallist = [(d_train, 'train'), (d_test, 'test')]

    num_rounds = 100
    bst = xgb.train(plst, d_train, num_rounds,
                    evallist, early_stopping_rounds=10)

    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    print 'save model to {}'.format(MODEL_PATH + '/model.bin')
    bst.save_model(MODEL_PATH + '/model.bin')

    pred = bst.predict(d_test)
    truth = test[label].values

    print 'test logloss: {}'.format(log_loss(truth, pred))


if __name__ == '__main__':
    main()
