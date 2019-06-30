from spark import get_spark, get_logger


def create_feature_map(fname, features):
    '''Write feature name for xgboost to map 'fn' -> feature name
        Args:
            fname(string): file name
            features(list): feature list
    '''
    with open(fname, 'w') as f:
        for i, feature in enumerate(features):
            f.write('{0}\t{1}\tq\n'.format(i, feature))


def create_feature_imp(fname, f_imp):
    '''Write feature importance file, and sort desc based on importance
        Args:
            fname(string): file name
            f_imp(dict): {feature_name(string): importance(numeric)}
    '''
    with open(fname, 'w') as f:
        for feature, imp in sorted(f_imp.items(), key=lambda v: v[1], reverse=True):
            f.write('{:20} {:.10f}\n'.format(feature, imp))


def print_summary(jmodel):
    '''Print train and valid summary for model
        Args:
            jmodel(ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier)
    '''
    # get spark and logger
    spark = get_spark(app_name="pyspark-xgb")
    logger = get_logger(spark, "app")

    train_summary = jmodel.summary().trainObjectiveHistory()
    valid_summary = jmodel.summary().validationObjectiveHistory()
    dataset_summary = [train_summary]
    dataset_name = ['train']
    for idx in range(valid_summary.size()):
        eval_name = valid_summary.apply(idx)._1()
        eval_summary = valid_summary.apply(idx)._2()
        dataset_name.append(eval_name)
        dataset_summary.append(eval_summary)

    stop_flg = False
    for round_idx, row in enumerate(zip(*dataset_summary), 1):
        printString = "{:6} ".format('[{}]'.format(round_idx))
        for idx, r in enumerate(row):
            if r == 0:
                stop_flg = True
                break
            printString += "{:5}\t{:10}\t".format(dataset_name[idx], r)

        if stop_flg is True:
            break
        logger.info(printString)
