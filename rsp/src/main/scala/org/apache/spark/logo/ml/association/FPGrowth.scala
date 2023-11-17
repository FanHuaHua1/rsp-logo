package org.apache.spark.logo.ml.association

import org.apache.spark.logo.etl.wrappers.BasicWrappers
import org.apache.spark.logo.strategy.global.MergeFPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.{Row, RspDataset}
import org.apache.spark.sql.RspContext.RspRDDFunc
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.Stream


object FPGrowth {

  def apply(minSupport: Double, itemsets: RDD[Row]) = {
    val Transac: RspRDD[Array[Array[Int]]] = etl(itemsets)
    val loResult: RspRDD[Stream[ItemSet]] = trainer(minSupport, Transac)
    val goResult: RspRDD[(String, Int)] = loResult.GO(MergeFPGrowth(_))
    goResult.foreach(println)
  }

  def etl(inputData: RDD[Row]): RspRDD[Array[Array[Int]]]= BasicWrappers.toMatrix(inputData)


  def trainer(minSupport: Double, itemsets: RspRDD[Array[Array[Int]]]): RspRDD[Stream[ItemSet]] = {
    itemsets.LO(trans => fpgrowth((trans.length * minSupport).toInt, trans))
  }

}