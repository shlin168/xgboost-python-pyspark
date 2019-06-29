#!/bin/sh
export SERVICE_HOME="$(cd "`dirname "$0"`"; pwd)"

# define your environment variable
export JAVA_HOME="/usr/lib/jvm/java"
export SPARK_HOME='/usr/local/spark-2.3.1-bin-hadoop2.6'

EXEC_PY=$1
JARS_PATH=${SERVICE_HOME}/jars/

spark-submit \
    --name 'spark xgb sample' \
    --master local \
    --jars ${JARS_PATH}/xgboost4j-spark-0.82.jar,${JARS_PATH}/xgboost4j-0.82.jar \
    ${SERVICE_HOME}/${EXEC_PY}
