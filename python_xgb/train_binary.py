import os
import pandas as pd
import numpy as np
import xgboost as xgb

from utils import create_feature_map, create_feature_imp


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

    params = {
        'eta': 0.1, 'eval_metric': 'logloss',
        'gamma': 0, 'max_depth': 5, 'min_child_weight': 1.0,
        'objective': 'binary:logistic', 'seed': 0,
        'verbosity': 0
    }
    plst = params.items()  # turn to tuple

    evallist = [(d_train, 'train'), (d_test, 'test')]

    num_rounds = 100
    bst = xgb.train(plst, d_train, num_rounds,
                    evallist, early_stopping_rounds=10)

    # write feature map and feature importance file
    imp_type = 'gain'
    feature_map_path = MODEL_PATH + '/feature.map'
    create_feature_map(feature_map_path, features)
    feature_imp_path = MODEL_PATH + '/feature.imp'
    create_feature_imp(feature_imp_path, bst.get_score(feature_map_path, imp_type))

    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    print 'save model to {}'.format(MODEL_PATH + '/model.bin')
    bst.save_model(MODEL_PATH + '/model.bin')

    # load model and predict
    bst = xgb.Booster({'nthread': 4})
    bst.load_model(MODEL_PATH + '/model.bin')
    pred = bst.predict(d_test)
    truth = test[target].values

    print 'test logloss: {}'.format(logloss(truth, pred))


if __name__ == '__main__':
    main()
