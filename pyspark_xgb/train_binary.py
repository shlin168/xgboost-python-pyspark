import os
import shutil
import traceback
import numpy as np

import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.wrapper import JavaWrapper

from spark import get_spark, get_logger
from schema import get_btrain_schema

# assert len(os.environ.get('JAVA_HOME')) != 0, 'JAVA_HOME not set'
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'

abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
PYSPARK_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
DATASET_PATH = PARENT_PROJ_PATH + '/dataset'
MODEL_PATH = PYSPARK_PROJ_PATH + '/binary_model'
LOCAL_MODEL_PATH = PARENT_PROJ_PATH + '/python_xgb/binary_model'


def udf_logloss(truth, pred, eps=1e-15):
    import math

    def logloss_(truth, pred):
        pred = eps if float(pred[1]) < eps else float(pred[1])
        return -(truth * math.log(pred) + (1 - truth) * math.log(1 - pred))
    return F.udf(logloss_, FloatType())(truth, pred)


def main():

    try:

        # init spark
        spark = get_spark(app_name="pyspark-xgb")

        # get logger
        logger = get_logger(spark, "app")

        # load data
        train = spark.read.schema(get_btrain_schema()).option('header', True).csv(
            DATASET_PATH + '/emp_train.csv')
        test = spark.read.schema(get_btrain_schema()).option('header', True).csv(
            DATASET_PATH + '/emp_test.csv')

        # preprocess
        LABEL = 'Attrition'
        FEATURES = 'features'
        features = [c for c in train.columns if c != LABEL]
        assembler = VectorAssembler(inputCols=features, outputCol=FEATURES)
        train = assembler.transform(train).select(FEATURES, LABEL)
        test = assembler.transform(test).select(FEATURES, LABEL)

        # training
        logger.info('training')
        xgb_params = {
            "eta": 0.1, "gamma": 0, "max_depth": 4,
            "num_round": 100, "num_early_stopping_rounds": 10,
            "num_workers": 1, "use_external_memory": False, "missing": np.nan,
            "eval_metric": "logloss", "min_child_weight": 1, "train_test_ratio": 0.8,
            "objective": "binary:logistic"
        }
        scala_map = spark._jvm.PythonUtils.toScalaMap(xgb_params)
        j = JavaWrapper._new_java_obj(
            "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier", scala_map) \
            .setFeaturesCol(FEATURES).setLabelCol(LABEL)
        jmodel = j.train(train._jdf)
        logger.info(jmodel.summary().toString())

        # get validation metric
        preds = jmodel.transform(test._jdf)
        pred = DataFrame(preds, spark)
        slogloss = pred.withColumn('log_loss', udf_logloss(LABEL, 'probability')) \
            .agg({"log_loss": "mean"}).collect()[0]['avg(log_loss)']
        logger.info('[xgboost4j] valid logloss: {}'.format(slogloss))

        # save or update model
        model_path = MODEL_PATH + '/model.bin'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            logger.info('model exist, rm old model')
        jw = JavaWrapper._new_java_obj(
            "ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel.XGBoostClassificationModelWriter",
            jmodel)
        jw.saveImpl(model_path)
        logger.info('save model to {}'.format(model_path))

        # [Optional] load model training by xgboost, predict and get validation metric
        local_model_path = LOCAL_MODEL_PATH + '/model.bin'
        if os.path.exists(local_model_path):
            logger.info('load model from {}'.format(local_model_path))
            scala_xgb = spark.sparkContext._jvm.ml.dmlc.xgboost4j.scala.XGBoost
            jbooster = scala_xgb.loadModel(local_model_path)

            # uid, num_class, booster
            xgb_cls_model = JavaWrapper._new_java_obj(
                "ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel",
                "xgbc", 2, jbooster)

            jpred = xgb_cls_model.transform(test._jdf)
            pred = DataFrame(jpred, spark)
            slogloss = pred.withColumn('log_loss', udf_logloss(LABEL, 'probability')) \
                .agg({"log_loss": "mean"}).collect()[0]['avg(log_loss)']
            logger.info('[xgboost] valid logloss: {}'.format(slogloss))
        else:
            logger.info(
                "local model is not exist, call python_xgb/train_binary.py to get the model "
                "and compare logloss between xgboost and xgboost4j"
            )

    except Exception:
        logger.error(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()
