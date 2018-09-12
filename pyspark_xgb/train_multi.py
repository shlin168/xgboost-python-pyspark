import os
import shutil
import traceback
import numpy as np

import pyspark.sql.functions as F
from pyspark.sql.types import FloatType
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.wrapper import JavaWrapper

from spark import get_spark, get_logger
from schema import get_mtrain_schema

assert len(os.environ.get('JAVA_HOME')) != 0, 'JAVA_HOME not set'
assert len(os.environ.get('SPARK_HOME')) != 0, 'SPARK_HOME not set'
assert not os.environ.get(
    'PYSPARK_SUBMIT_ARGS'), 'PYSPARK_SUBMIT_ARGS should not be set'

abspath = os.path.abspath(__file__)
PARENT_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-2])
PYSPARK_PROJ_PATH = '/'.join(abspath.split(os.sep)[:-1])
DATASET_PATH = PARENT_PROJ_PATH + '/dataset'
MODEL_PATH = PYSPARK_PROJ_PATH + '/multi_model'


def udf_logloss(truth, pred, n_class, eps=1e-15):
    import math

    def logloss_(truth, pred):
        truth_array = [0] * n_class
        truth_array[int(truth)] = 1
        pred = [eps if p < eps else float(p) for p in pred]
        logloss = sum([t * math.log(p) if t == 1 else (1 - t) *
                       math.log(1 - p) for t, p in zip(truth_array, pred)])
        return -logloss
    return F.udf(logloss_, FloatType())(truth, pred)


def main():

    try:

        # init spark
        spark = get_spark(app_name="pyspark-xgb")

        # get logger
        logger = get_logger(spark, "app")

        # load data
        df = spark.read.csv(DATASET_PATH + "/iris.data", get_mtrain_schema())

        # preprocess
        LABEL = 'label'
        FEATURES = 'features'
        N_CLASS = 3
        features = [c for c in df.columns if c != "class"]
        assembler = VectorAssembler(inputCols=features, outputCol='features')
        strIdxer = StringIndexer(inputCol="class", outputCol=LABEL)
        pipeline = Pipeline(stages=[assembler, strIdxer])
        df = pipeline.fit(df).transform(df).select(FEATURES, LABEL)
        train, test = df.randomSplit([0.8, 0.2])

        # training
        logger.info('training')
        xgb_params = {
            "eta": 0.1, "gamma": 0, "max_depth": 4,
            "num_round": 100, "num_early_stopping_rounds": 10,
            "num_workers": 1, "use_external_memory": False, "missing": np.nan,
            "num_class": 3, "eval_metric": "mlogloss",
            "min_child_weight": 1, "train_test_ratio": 0.8,
            "objective": "multi:softprob"
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
        slogloss = pred.withColumn('log_loss', udf_logloss('label', 'probability', N_CLASS)) \
            .agg({"log_loss": "mean"}).collect()[0]['avg(log_loss)']
        logger.info('valid logloss: {}'.format(slogloss))

        # save or update model
        model_path = MODEL_PATH + '/model.bin'
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            logger.info('model exist, rm old model')
        jmodel.save(model_path)
        logger.info('save model to {}'.format(model_path))

    except Exception:
        logger.error(traceback.print_exc())

    finally:
        # stop spark
        spark.stop()


if __name__ == '__main__':
    main()
