# xgb_demo
xgboost in python and pyspark (using py4j to call jvm-packages)

## how to run

### build images from docker file (~3GB)
> it takes some time to build the images ...
```
cd docker
docker build -t xgb:latest . --no-cache
```

### start docker container using images, go to project directory
```
docker run -i -t xgb:latest /bin/bash
cd xgb_demo
```

### run xgboost
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

> TODO: wrong logloss in multiple class prediction
