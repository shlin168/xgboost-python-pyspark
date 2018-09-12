import os
import pandas as pd
import numpy as np
import xgboost as xgb


abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
LOCAL_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
DATASET_PATH = PARENT_PROJ_PATH + '/dataset'
MODEL_PATH = LOCAL_PROJ_PATH + '/binary_model'


def logloss(truth, pred):
    return np.mean(-(truth * np.log(pred) + (1 - truth) * np.log(1 - pred)))


def main():
    train = pd.read_csv(DATASET_PATH + '/emp_train.csv')
    test = pd.read_csv(DATASET_PATH + '/emp_test.csv')

    target = 'Attrition'
    features = [c for c in train.columns if c != target]

    d_train = xgb.DMatrix(train[features], train[target], missing=np.nan)
    d_test = xgb.DMatrix(test[features], test[target], missing=np.nan)

    params = {'max_depth': 10, 'min_child_weight': 3.0, 'eval_metric': 'auc',
              'seed': 0, 'objective': 'binary:logistic', 'eta': 0.1}
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
    truth = test[target].values

    print 'test logloss: {}'.format(logloss(truth, pred))


if __name__ == '__main__':
    main()
