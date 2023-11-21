package org.apache.spark.logo

import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.{Row, RspDataset}
import org.apache.spark.sql.RspContext.RspRDDFunc

import scala.util.Random

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/21 15:36
 * @desc
 */
object Utils {

  def transform(rdd: RspRDD[Row], trainParNums: Int, predictParNums: Int):(RspRDD[Row], RspRDD[Row]) = {
    val rddPartitions: Int = rdd.getNumPartitions
    if (trainParNums > rddPartitions || (trainParNums + predictParNums) > rddPartitions)
      throw new IllegalArgumentException("input partitions number is larger than rdd partitions number.")
    val trainRdd = rdd.getSubPartitions(trainParNums)
    var predictRdd: RspRDD[Row] = null
    if (predictParNums > 0) {
      predictRdd = rdd.getSubPartitions(predictParNums)
    }
    (trainRdd, predictRdd)
  }

  def generateShuffleArray(sizes: Array[Int], shuffleMaxNum: Int): Array[(Int, Array[Int])] = {
    val partitionsList = List.range(0, shuffleMaxNum)
    if (sizes.length > 0) {
      sizes.map(s => (s, Random.shuffle(partitionsList).toArray))
    } else {
      Array((0, Array()))
    }
  }

}
