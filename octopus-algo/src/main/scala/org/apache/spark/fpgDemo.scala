package org.apache.spark

import org.apache.spark.logo.ml.association.{fpgBroadcast, fpgVote}
import org.apache.spark.sql.RspContext.SparkSessionFunc
import org.apache.spark.sql.SparkSession

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/15 14:20
 * @desc
 */
object fpgDemo {
  def main(args: Array[String]): Unit = {
    val sparkconf = new SparkConf().setAppName("Test_Smile").setMaster("local[*]")
    val spark = SparkSession
      .builder()
      .config(sparkconf)
      .getOrCreate()
    println("------------环境配置成功---------------")
    val transaction= spark.rspRead.text("E:/expdatas/datas/items_40.txt")
    fpgVote(0.15 * transaction.count() / transaction.rdd.getNumPartitions, transaction.rdd)
  }
}
