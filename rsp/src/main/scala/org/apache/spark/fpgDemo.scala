package org.apache.spark

import org.apache.spark.logo.ml.association.fpg
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

    val transaction= spark.rspRead.parquet("datas/transaction_dataset_30.parquet")

    spark.rspRead.parquet("datas/transaction_dataset_30.parquet")
    println("------数据加载成功-------")

    fpg(0.21, transaction.rdd)


  }

}
