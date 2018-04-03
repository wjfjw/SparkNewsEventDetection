#!/bin/bash
spark-submit \
--master spark://wjf.wjf:7077 \
--class priv.wjf.project.SparkNewsEventDetection.App_experiment \
--executor-memory 4608m \
/home/wjf/JavaProject/SparkNewsEventDetection/target/SparkNewsEventDetection-0.0.1-SNAPSHOT.jar
