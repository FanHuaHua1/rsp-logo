package org.apache.spark.logo.ml.association

import org.apache.spark.logo.etl.wrappers.BasicWrappers
import org.apache.spark.logo.strategy.global.association.{Broadcast, Vote}
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.RspContext.RspRDDFunc
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.{Collectors, Stream}
import scala.collection.JavaConverters._
import scala.collection.mutable


class BroadcastFPGrowth(var minSupport: Double = 0.15, var itemsets: RDD[Row]) {
  var rowCount = 0;

  def this() = this(0, null)

  def etl(inputData: RDD[Row]): RspRDD[Array[Array[Int]]]= BasicWrappers.toMatrix(inputData)
  def etlText(inputData: RDD[Row]): RspRDD[Array[Int]]= BasicWrappers.txtToStrinWithoutGlom(inputData)

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

  def runText(saveModel: Boolean, saveModelPath: String) = {
    val Transac: RspRDD[Array[Int]] = etlText(itemsets)
    val glomTransac = new RspRDD(Transac.glom())
    rowCount = glomTransac.count().toInt
    val loResult: RspRDD[ItemSet] = trainer(minSupport, glomTransac)
    val goResult: RspRDD[(String, Double)] = Broadcast(Transac, loResult, rowCount, minSupport)
    if (saveModel) {
      goResult.saveAsObjectFile(saveModelPath)
    }
    val res: Array[(String, Double)] = goResult.collect()
    println("集成结果:")
    println(res.mkString("\n"))
  }

}

object BroadcastFPGrowth {
  def apply(minSupport: Double, itemsets: RDD[Row]) = {
    new BroadcastFPGrowth(minSupport, itemsets).runText(false, null)
  }
}
