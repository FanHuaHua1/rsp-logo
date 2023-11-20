package org.apache.spark.logo.ml.association

import org.apache.spark.logo.etl.wrappers.BasicWrappers
import org.apache.spark.logo.strategy.global.association.Vote
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.RspContext.RspRDDFunc
import smile.association.{ItemSet, fpgrowth}

import java.util.stream.{Collectors, Stream}
import scala.collection.JavaConverters._
import scala.collection.mutable


class VoteFPGrowth(var minSupport: Double = 0.15, var itemsets: RDD[Row]) {
  def this() = this(0, null)

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

  def run(saveModel: Boolean, saveModelPath: String) = {
    val Transac: RspRDD[Array[Array[Int]]] = etl(itemsets)
    val loResult: RspRDD[ItemSet] = trainer(minSupport, Transac)
    val goResult: RspRDD[(String, Int)] = loResult.GO(Vote(_))
    if(saveModel){
      goResult.saveAsObjectFile(saveModelPath)
    }
    goResult.foreach(println)
  }

  def etlText(inputData: RDD[Row]): RspRDD[Array[Array[Int]]]= BasicWrappers.txtToString(inputData)


  def runText(saveModel: Boolean, saveModelPath: String) = {
    val Transac: RspRDD[Array[Array[Int]]] = etlText(itemsets)
    val loResult: RspRDD[ItemSet] = trainer(minSupport, Transac)
    val goResult: RspRDD[(String, Int)] = loResult.GO(Vote(_))
    if (saveModel) {
      goResult.saveAsObjectFile(saveModelPath)
    }
    val res: Array[(String, Int)] = goResult.collect()
    println("集成结果:")
    println(res.mkString("\n"))
  }

}

object VoteFPGrowth {
  def apply(minSupport: Double, itemsets: RDD[Row]) = {
    new VoteFPGrowth(minSupport, itemsets).runText(false, null)
  }
}
