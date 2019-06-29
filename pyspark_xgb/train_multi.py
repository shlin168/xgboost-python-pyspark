import os
import shutil
import traceback
import numpy as np

import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.wrapper import JavaWrapper

from spark import get_spark, get_logger
from schema import get_mtrain_schema


# assert len(os.environ.get('JAVA_HOME')) != 0, 'JAVA_HOME not set'
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'

abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
PYSPARK_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
DATASET_PATH = PARENT_PROJ_PATH + '/dataset'
MODEL_PATH = PYSPARK_PROJ_PATH + '/multi_model'
LOCAL_MODEL_PATH = PARENT_PROJ_PATH + '/python_xgb/multi_model'


def udf_logloss(truth, pred, n_class, eps=1e-15):
    # have been validated using logloss in sklearn metric
    import math

    def logloss_(truth, pred):
        truth_array = [0] * n_class
        truth_array[int(truth)] = 1
        pred = [eps if p < eps else float(p) for p in pred]
        logloss = sum([t * math.log(p) for t, p in zip(truth_array, pred)])
        return -logloss
    return F.udf(logloss_, FloatType())(truth, pred)


def main():

    try:

        # init spark
        spark = get_spark(app_name="pyspark-xgb")

        # get logger
        logger = get_logger(spark, "app")

        # load data
        train = spark.read.csv(DATASET_PATH + "/iris_train.csv",
                    get_mtrain_schema(),
                    header=True)
        test = spark.read.csv(DATASET_PATH + "/iris_test.csv",
                    get_mtrain_schema(),
                    header=True)

        # preprocess
        # get label encode result from csv, since StringIndexer will get different result
        STR_LABEL = 'class'
        LABEL = 'label'
        FEATURES = 'features'
        N_CLASS = 3
        features = [c for c in train.columns if c not in [STR_LABEL, LABEL]]
        assembler = VectorAssembler(inputCols=features, outputCol=FEATURES)
        pipeline = Pipeline(stages=[assembler])
        preprocess = pipeline.fit(train)
        train = preprocess.transform(train).select(FEATURES, LABEL)
        test = preprocess.transform(test).select(FEATURES, LABEL)

        # set param map
        xgb_params = {
            "eta": 0.1, "eval_metric": "mlogloss",
            "gamma": 0, "max_depth": 5, "min_child_weight": 1.0,
            "objective": "multi:softprob", "seed": 0,
            "num_class": N_CLASS,
            # xgboost4j only
            "num_round": 100, "num_early_stopping_rounds": 10,
            "maximize_evaluation_metrics": False,
            "num_workers": 1, "use_external_memory": False,
            "missing": np.nan,
        }
        scala_map = spark._jvm.PythonUtils.toScalaMap(xgb_params)

        # set evaluation set
        eval_set = {'eval': test._jdf}
        scala_eval_set = spark._jvm.PythonUtils.toScalaMap(eval_set)

        logger.info('training')
        j = JavaWrapper._new_java_obj(
            "ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier", scala_map) \
            .setFeaturesCol(FEATURES).setLabelCol(LABEL) \
            .setEvalSets(scala_eval_set)
        jmodel = j.train(train._jdf)
        logger.info(jmodel.summary().toString())

        # get validation metric
        preds = jmodel.transform(test._jdf)
        pred = DataFrame(preds, spark)
        slogloss = pred.withColumn('log_loss', udf_logloss(LABEL, 'probability', N_CLASS)) \
            .agg({"log_loss": "mean"}).collect()[0]['avg(log_loss)']
        logger.info('[xgboost4j] valid logloss: {}'.format(slogloss))

        # save or update model
        model_path = MODEL_PATH + '/model.bin'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            logger.info('model exist, rm old model')
        jmodel.save(model_path)
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
                "xgbc", N_CLASS, jbooster)

            jpred = xgb_cls_model.transform(test._jdf)
            pred = DataFrame(jpred, spark)
            slogloss = pred.withColumn('log_loss', udf_logloss(LABEL, 'probability', N_CLASS)) \
                .agg({"log_loss": "mean"}).collect()[0]['avg(log_loss)']
            logger.info('[xgboost] valid logloss: {}'.format(slogloss))
        else:
            logger.info(
                "local model is not exist, call python_xgb/train_multi.py to get the model "
                "and compare logloss between xgboost and xgboost4j"
            )

    except Exception:
        logger.error(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()
