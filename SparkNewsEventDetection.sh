#!/bin/bash
spark-submit \
--master local \
--class priv.wjf.project.SparkNewsEventDetection.App \
--executor-memory 6g \
/home/wjf/JavaProject/SparkNewsEventDetection/target/SparkNewsEventDetection-0.0.1-SNAPSHOT.jar
