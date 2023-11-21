#! /bin/bash

function join { local IFS="$1"; shift; echo "$*"; }

appName=$(join - $*)

echo submit $appName

root=$PWD

# jars=$(join , `ls $PWD/targets/*jar`)
algojar=$root/target/spark-rsp_2.11-2.4.0-jar-with-dependencies.jar

submitcmd="$SPARK_HOME/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--class org.apache.spark.logo.App \
--conf spark.yarn.maxAppAttempts=0 \
--name $appName \
$algojar $*"

if [ ! -d "logs" ]
then
    mkdir logs
fi

echo `date "+%Y-%m-%dT%T"` $submitcmd >> logs/submit-commands-history.log

$SPARK_HOME/bin/spark-submit \
--master yarn \
--deploy-mode cluster \
--class org.apache.spark.logo.App \
--conf spark.yarn.maxAppAttempts=0 \
--conf spark.task.maxFailures=1 \
--name $appName \
$algojar $*
