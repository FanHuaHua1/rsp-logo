package org.apache.spark.logo.ml.association

import org.apache.spark.logo.etl.wrappers.BasicWrappers
import org.apache.spark.logo.strategy.global.association.Vote
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.{Row}
import org.apache.spark.sql.RspContext.RspRDDFunc
import smile.association.{ItemSet, fpgrowth}
import scala.collection.JavaConverters._
import java.util.stream.{Collectors, Stream}
import scala.collection.mutable


object FPGrowth {

  def apply(minSupport: Double, itemsets: RDD[Row]) = {
    val Transac: RspRDD[Array[Array[Int]]] = etl(itemsets)
    val loResult: RspRDD[ItemSet] = trainer(minSupport, Transac)
    val goResult: RspRDD[(String, Int)] = loResult.GO(Vote(_))
    goResult.foreach(println)
  }

  def etl(inputData: RDD[Row]): RspRDD[Array[Array[Int]]]= BasicWrappers.toMatrix(inputData)


  def trainer(minSupport: Double, itemsets: RspRDD[Array[Array[Int]]]): RspRDD[ItemSet] = {
    val valueStreamRdd: RspRDD[Stream[ItemSet]] = itemsets.LO(trans => fpgrowth((trans.length * minSupport).toInt, trans))
    val valueSetRdd: RDD[ItemSet] = valueStreamRdd.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
      //迭代器里只有一个stream.Stream[ItemSet]
      val elem: Stream[ItemSet] = stream.next()
      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
      buf.iterator
    })
    new RspRDD(valueSetRdd)
  }

}