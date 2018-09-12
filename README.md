# xgb_demo
xgboost in python and pyspark (using py4j to call jvm-packages)

## how to run

### build environment from docker file
```
docker build -t xgb:dockerfile -f docker/DockerFile .
```

### python
> python version 2.7
* binary logistic
```
python python_xgb/train_binary.py
```
* multi classification
```
python python_xgb/train_multi.py
```

### pyspark (py4j to call function in xgboost jvm-packages)
> spark version 2.3.+
* binary logistic
```
pyspark_xgb/start.sh train_binary.py
```
* multi classification
```
pyspark_xgb/start.sh train_multi.py
```
