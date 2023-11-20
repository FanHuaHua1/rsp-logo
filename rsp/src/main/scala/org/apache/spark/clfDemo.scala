package org.apache.spark

import org.apache.spark.logo.ml.classification.{LogisticRegression, dt, lr, rf}
import org.apache.spark.sql.RspContext.SparkSessionFunc
import org.apache.spark.sql.{Row, RspDataset, SparkSession}

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/15 14:20
 * @desc
 */
object clfDemo {
  def main(args: Array[String]): Unit = {
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    val filePath = if(args.length > 0) args(0) else "datas/classification_50_2_0.54_5_64M.parquet"
    val frame: RspDataset[Row] = spark.rspRead.parquet(filePath)

    //val value: RDD[(LogisticRegression, Double)] = lr(frame.rdd, 4, 1, true, 0.05, 500, 0.1)
    val value = new LogisticRegression(frame.rdd, null, false, 0.05, 500, 0.1).run(true, "modules/ob22", false)

    //val value: RDD[(RandomForest, Double)] = rf(frame.rdd, 4, 1, true, 0.05, 20, 5)
    //val value: RDD[(DecisionTree, Double)] = dt(frame.rdd, 4, 1)

    value.collect()
    println("============" + value.getNumPartitions)
    value.foreach(f => println("===========" + f._2))
//    value.map(_._1).saveAsObjectFile("modules/ob")
//    spark.close()
//    val sc = new SparkContext(sparkconf)
//    val value1: RDD[DecisionTree] = sc.objectFile("modules/ob")
//    value1.foreach(f => println(f))
  }
}
