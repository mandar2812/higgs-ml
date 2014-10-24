#! /bin/bash

spark-submit --class org.mandar.analysis.higgsml.svmTrial \
    --master yarn-cluster \
    --num-executors 3 \
    --driver-memory 4g \
    --executor-memory 2g \
    --executor-cores 1 \
    target/higgs-ml-1.0-SNAPSHOT.jar \
    10
