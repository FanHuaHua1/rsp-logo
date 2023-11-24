package org.apache.spark.logo.ml.association

import org.apache.spark.logo.Utils
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.{Row, RspDataset, SparkSession}

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.util.Random

object Entrypoint {

  def onArgs(spark: SparkSession, args: Array[String]): Unit = {
    printf("onArgs: %s\n", args.reduce((a, b) => a + " " + b))
    args(0) match {
      case "logoShuffle" => fitLogoShuffle(spark, args)
      case _ => printf("Unknown type: %s\n", args(0))
    }
  }

  def fitLogoShuffle(spark: SparkSession, args: Array[String]): Unit = {
    if (args.length <= 5) {
      logoShuffle(spark, args(1), args(2), args(3), args(4).toBoolean,
        0.15, 1.0, Array[Int] {
          1
        })
    } else {
      logoShuffle(
        spark,
        args(1), args(2), args(3), args(4).toBoolean,
        args(5).toDouble, args(6).toDouble, args.slice(7, args.length).map(_.toInt))
    }

  }

  def logoShuffle(spark: SparkSession,
                      algo: String,
                      sourceFile: String,
                      modelPath: String,
                      isCrossDomain: Boolean,
                      minsup:Double,
                      sub:Double,
                      sizes: Array[Int]): Unit = {
    val rdf: RspDataset[Row] = spark.rspRead.text(sourceFile)
    val jobs: Array[(Int, Array[Int])] = Utils.generateShuffleArray(sizes, rdf.rdd.getNumPartitions)
    val fpgAlgo = algo match {
      case "Vote" => new VoteFPGrowth()
      //case "Broadcast" => new BroadcastFPGrowth()
    }
    for ((totalSize, partitions) <- jobs) {
      val size = Math.ceil(totalSize * sub).toInt
      var trainName = "train(size=%d)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      print("%s start" + trainName)
      val trainRdd: RspRDD[Row] = rdf.rdd.getSubPartitions(partitions.slice(0, size))
      val l: Long = trainRdd.count()
      print("l: " + l )
      //val savePath = modelPath + "_" + totalSize
      val savePath = modelPath
      //这里是把minsup转化为对应某个分区的比例，
      // 比如10个块，原本是0.15的支持度，l是总数据条目，那么折算到每个块的（支持度对应的条数）就是 0.015 * l
      fpgAlgo.minSupport = (l * minsup / size).toInt
      fpgAlgo.itemsets = trainRdd
      fpgAlgo.runText(isCrossDomain, true, savePath)
      //SmileFPGrowth.runv(value, vote, (l * minsup / size).toInt, "/user/caimeng/fpg" + System.currentTimeMillis() + sizex * 100 + "_" + minsup, inputPath)
    }
  }

//  def logoShuffleBroadCast(spark: SparkSession,
//                      sourceFile: String,
//                      modelPath: String,
//                      minsup: Double,
//                      sub: Double,
//                      sizes: Array[Int]): Unit = {
//    val rdf: RspDataset[Row] = spark.rspRead.text(sourceFile)
//    val jobs: Array[(Int, Array[Int])] = Utils.generateShuffleArray(sizes, rdf.rdd.getNumPartitions)
//    val fpg = new BroadcastFPGrowth()
//    for ((totalSize, partitions) <- jobs) {
//      val size = Math.ceil(totalSize * sub).toInt
//      var trainName = "train(size=%d)".format(size)
//      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
//      val trainRdd: RspRDD[Row] = rdf.rdd.getSubPartitions(partitions.slice(0, size))
//      val l: Long = trainRdd.count()
//      val savePath = modelPath + "_" + totalSize
//      //这里是把minsup转化为对应某个分区的比例，
//      // 比如10个块，原本是0.15的支持度，l是总数据条目，那么折算到每个块的（支持度对应的条数）就是 0.015 * l
//      fpg.minSupport = (l * minsup / size).toInt
//      fpg.itemsets = trainRdd
//      fpg.runText(true, savePath)
//      //SmileFPGrowth.runv(value, vote, (l * minsup / size).toInt, "/user/caimeng/fpg" + System.currentTimeMillis() + sizex * 100 + "_" + minsup, inputPath)
//    }
//  }

//  def logoShuffleBroadCast(spark: SparkSession,
//                   sourceFile: String,
//                   minsup:Double,
//                   blockSizes: Array[Int]): Unit = {
//    for (size <- blockSizes) {
//      val block = (size / 5 * 10).toInt
//      val sampleBlock = Math.ceil(block * 0.05).toInt
//      val trainRdd = spark.rspRead.text("Items_" + size + "W_" + block + "_5K_RSP.txt")
//      val partitionsList = List.range(0, trainRdd.rdd.getNumPartitions)
//      val array: Array[Int] = Random.shuffle(partitionsList).toArray
//      val value1: RspRDD[Row] = trainRdd.rdd.getSubPartitions(array.slice(0, sampleBlock))
//      var trainName = "train(size=%d)".format(size)
//      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
//      val value: RDD[String] = value1.map((f: Row) => f.mkString(" "))
//      println(value.first())
//      println(trainName)
//      val l: Long = value.count()
//      SmileFPGrowth.runb(value, l, minsup, (l * minsup / sampleBlock).toInt, "/user/caimeng/xxx1size_" + size + "_minsup_0.15_new_50_")
//    }
//  }

}
