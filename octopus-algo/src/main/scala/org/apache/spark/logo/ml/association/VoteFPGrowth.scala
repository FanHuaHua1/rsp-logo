package org.apache.spark.logo.ml.association

import org.apache.spark.logo.etl.wrappers.BasicWrappers
import org.apache.spark.logo.strategy.global.association.Vote
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.RspContext.RspRDDFunc
import smile.association.{ItemSet, fpgrowth}

import java.util.function.Predicate
import java.util.stream.{Collectors, Stream}
import scala.collection.JavaConverters._
import scala.collection.mutable


class VoteFPGrowth(var minSupport: Double = 0.15, var itemsets: RDD[Row]) extends AssociationJob{
  def this() = this(0, null)

  def etl(inputData: RDD[Row]): RspRDD[Array[Array[Int]]]= BasicWrappers.toMatrix(inputData)

  def trainer(minSupport: Double, itemsets: RspRDD[Array[Array[Int]]]): RspRDD[(String, Int)] = {
    val valueStreamRdd: RspRDD[Stream[ItemSet]] = itemsets.LO(trans => fpgrowth(minSupport.toInt, trans))
    val valueSetRdd: RDD[(String, Int)] = valueStreamRdd.mapPartitions((stream: Iterator[Stream[ItemSet]]) => {
      //迭代器里只有一个stream.Stream[ItemSet]
      //val elem: Stream[ItemSet] = stream.next()
      val elem: Stream[ItemSet] = stream.next().filter(new Predicate[ItemSet]() {
        override def test(t: ItemSet) = t.items.length > 1
      })
      val buf: mutable.Buffer[ItemSet] = elem.collect(Collectors.toList[ItemSet]).asScala
      val kv: mutable.Buffer[(String, Int)] = buf.map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
      kv.iterator
    })
    new RspRDD(valueSetRdd)
  }

  def run(saveModel: Boolean, saveModelPath: String) = {
    val Transac: RspRDD[Array[Array[Int]]] = etl(itemsets)
    val loResult: RspRDD[(String, Int)] = trainer(minSupport, Transac)
    val goResult: RspRDD[(String, Int)] = loResult.GO(Vote(_))
    if(saveModel){
      goResult.saveAsObjectFile(saveModelPath)
    }
    goResult.foreach(println)
  }

  def etlText(inputData: RDD[Row]): RspRDD[Array[Array[Int]]]= BasicWrappers.txtToString(inputData)

  def runText(isCrossDomain:Boolean, saveModel: Boolean, saveModelPath: String) = {
    val Transac: RspRDD[Array[Array[Int]]] = etlText(itemsets)
    println("minsupport:" + minSupport)
    val loResult: RspRDD[(String, Int)] = trainer(minSupport, Transac)
    if(isCrossDomain){
      println("save lo result directly for job crossdomain.")
      println("lo result partition count:" + loResult.partitions.length)
      loResult.saveAsObjectFile(saveModelPath)
    } else {
      println("Not cross domain. Go integeration start.")
      val goResult: RspRDD[(String, Int)] = loResult.GO(Vote(_))
      if (saveModel) {
        goResult.saveAsObjectFile(saveModelPath)
      }
      val res: Array[(String, Int)] = goResult.collect()
      println("result count: " + res.length)
    }
  }
}

object VoteFPGrowth {
  def apply(minSupport: Double, itemsets: RDD[Row], isCrossDomain:Boolean = false, saveModel: Boolean = false, saveModelPath: String = null) = {
    new VoteFPGrowth(minSupport, itemsets).runText(isCrossDomain, saveModel, saveModelPath)
  }
}
