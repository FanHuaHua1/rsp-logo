package org.apache.spark

import org.apache.spark.logo.ml.clustering.BisectingKMeans
import org.apache.spark.sql.RspContext.SparkSessionFunc
import org.apache.spark.sql.{Row, RspDataset, SparkSession}

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/15 14:20
 * @desc
 */
object cltDemo {
  def main(args: Array[String]): Unit = {
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")

    //val frame: RspDataset[Row] = spark.rspRead.parquet(args(0))
    val frame: RspDataset[Row] = spark.rspRead.parquet("datas/1118")

    //val array: Array[Array[Double]] = KMeans(frame.rdd, 2, 0)
    val array: Array[Array[Double]] = BisectingKMeans(frame.rdd, 2, 0)
    array.foreach(f => println("===========" + f.mkString(",")))
//    value.map(_._1).saveAsObjectFile("modules/ob")
//    spark.close()
//    val sc = new SparkContext(sparkconf)
//    val value1: RDD[DecisionTree] = sc.objectFile("modules/ob")
//    value1.foreach(f => println(f))

  }

}
