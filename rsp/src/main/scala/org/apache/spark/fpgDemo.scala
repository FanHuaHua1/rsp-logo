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

    //val transaction= spark.rspRead.parquet("datas/transaction_dataset_30.parquet")
    val transaction= spark.rspRead.text("datas/items_40.txt")
    println("------数据加载成功-------")
    //fpgVote(0.2, transaction.rdd)
    fpgBroadcast(0.2, transaction.rdd)


  }

}
