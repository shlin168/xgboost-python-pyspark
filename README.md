# xgboost in python and pyspark
xgboost in python and pyspark (using py4j to call jvm-packages)<br/>
xgboost4j version: `0.82`
> TODO: xgboost4j is not the latest version since 0.90 only supports `python3` and `spark 2.4`

## how to set environment (without docker)
1. download `xgboost4j-0.82` jar files from
[xgboost-jars](https://github.com/criteo-forks/xgboost-jars/releases)
2. copy to `pyspark_xgb/jars`
3. rename to `xgboost4j-0.82.jar` and `xgboost4j-spark-0.82.jar` respectively
4. set your `SPARK_HOME` and `JAVA_HOME` in `pyspark/start.sh`
5. [opt] change spark-submit parameters if needed

## run xgboost
> python version 2.7
* binary logistic
```
python python_xgb/train_binary.py
```
* multi classification
```
python python_xgb/train_multi.py
```

### run xgboost4j (py4j to call function in xgboost jvm-packages)
> spark version 2.3.*
* binary logistic
```
pyspark_xgb/start.sh train_binary.py
```
* multi classification
```
pyspark_xgb/start.sh train_multi.py
```

# Appendix
run the program within docker

## how to set environment (docker)
### build images from docker file (~3GB)
> it takes some time to build the images ...
```
cd docker
docker build -t xgb:latest . --no-cache
```
### start docker container using images, go to project directory
```
docker run -i -t xgb:latest /bin/bash
cd xgboost-python-pyspark
```
