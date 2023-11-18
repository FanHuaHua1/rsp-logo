package org.apache.spark.logo.strategy.global.association

import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import smile.association.ItemSet
import java.util
import java.util.stream.{Collectors, Stream}
import scala.collection.JavaConverters.asScalaBufferConverter
import scala.collection.mutable

/**
 * @Author Lingxiang Zhao
 * @Date 2023/11/17 20:13
 * @desc
 */
object Broadcast {
  def apply(transaction: RDD[Array[Int]], itemSetRDD: RspRDD[ItemSet], count: Long, elem: Double): RspRDD[(String, Double)] = {
    println("正在对本地频繁项集建模进行集成.....")
    //用于广播
    val broadcastList2 = itemSetRDD
      .filter((item: ItemSet) => item.items.length > 1)
      .map((item: ItemSet) => item.items.toList.sorted.mkString(","))
      .distinct()
      .collect()

    broadcastList2.foreach(println)
    val broadcastList: Array[List[Int]] = broadcastList2.map(_.split(",").map(_.toInt).toList)
    val value1: RDD[(String, Int)] = transaction.mapPartitions((arr: Iterator[Array[Int]]) => {
      val temp: Array[List[Int]] = util.Arrays.copyOf(broadcastList, broadcastList.length) //广播数组
      val set: Array[Set[Int]] = arr.map(_.toSet).toArray
      val partitionRes: Array[(String, Int)] = temp.map(items => { //List[Int]
        var count = 0
        for (orginalData <- set) { //List[Int]
          var b = true
          for (item <- items if b) { //Int
            if (!orginalData.contains(item)) {
              b = false
            }
          }
          if (b) {
            count = count + 1
          }
        }
        (items.mkString("{", ",", "}"), count)
      })
      partitionRes.iterator
    })
    val itemSetWithFreqAndCount = value1.reduceByKey(_ + _).map(x => (x._1, x._2 * 1.0 / count)).filter(_._2 >= elem)

    new RspRDD(itemSetWithFreqAndCount)
  }
}
