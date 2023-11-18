package org.apache.spark.logo.strategy.global.association

import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import smile.association.ItemSet

import java.util.stream.{Collectors, Stream}
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable
/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/15 10:50
 * @desc
 */
object Vote {
  def apply(itemSetRDD: RspRDD[ItemSet]): RspRDD[(String, Int)] = {
    println("正在对本地频繁项集建模进行集成.....")

    val modelNum = itemSetRDD.partitions.length
    itemSetRDD.map(f => (f.items, f.support)).saveAsObjectFile("modules/ob2")
    //将Itemset对象转成（频繁项集，出现次数）的KV对
    val itemSetWithFreq: RDD[(String, Int)] = itemSetRDD
      .filter(item => item.items.length > 1)
      .map((item: ItemSet) => (item.items.toList.sorted.mkString("{", ",", "}"), item.support))
    val itemSetWithFreqAndCount: RDD[(String, Int)] = itemSetWithFreq.map(elem => {
      (elem._1, (elem._2, 1))
    }).reduceByKey(
      (x, y) => (x._1 + y._1, x._2 + y._2)
    ).filter(
      elem => {
        elem._2._2 >= modelNum * 0.5
      }
    ).map(
      x =>
        (x._1, x._2._1 / x._2._2)
    )
    new RspRDD(itemSetWithFreqAndCount)
  }
}
