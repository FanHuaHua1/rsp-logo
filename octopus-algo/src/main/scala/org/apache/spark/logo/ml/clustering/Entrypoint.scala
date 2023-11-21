package org.apache.spark.logo.ml.clustering

import org.apache.spark.logo.Utils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.collection.mutable.WrappedArray

object Entrypoint {
  /**
   *
   * @param spark
   * @param args: Array[String], spark|logo commands...
   */
  def onArgs(spark: SparkSession, args: Array[String]): Unit = {
    args(0) match {
      case "logoShuffle" => logoShuffle(
        spark,
        args(1), args(2), args(3), args(4),
        args(5).toDouble, args(6).toInt,
        args(7).toDouble,
        args.slice(8, args.length).map(_.toInt)
      )
      case _ => printf("Unknown type: %s\n", args(0))
    }
  }

  def readCenters(spark: SparkSession, centersFile: String): Map[String, Array[Array[Double]]] = {
    val df = spark.read.parquet(centersFile)
    df.rdd.map(r => (
      r.getString(0),
      r.get(1).asInstanceOf[WrappedArray[DenseVector]].toArray.map(_.toArray)
    )).collect().toMap
  }

  /**
   *
   * @param spark
   * @param algo 算法：kmeans|bisectingkmeans
   * @param sourceFile 数据原文件
   * @param centersFile 中心点文件
   * @param subs rsp取块比例
   * @param tests 测试集块数，0表示跳过测试集
   * @param partitionsUnit 单位数据块数, 计算取块数 = (size * partitionsUnit).toInt()
   * @param sizes Array[Int], 训练数据集大小
   */
  def logoShuffle(spark: SparkSession,
                  algo: String,
                  sourceFile: String,
                  modelPath: String,
                  centersFile: String,
                  subs: Double,
                  tests: Int,
                  partitionsUnit: Double,
                  sizes: Array[Int]): Unit = {

    printf("fitLogo: algorithm = %s\n", algo)
    printf("fitLogo: sourceFile = %s\n", sourceFile)
    var rdf = spark.rspRead.parquet(sourceFile)
    var fileList = centersFile.split(":")
    val centerMap = readCenters(spark, fileList(0))
    val key: String = {
      if(fileList.length > 1) {
        fileList(1)
      }else{
        val sourceFilePath: Array[String] = sourceFile.split("/")
        sourceFilePath(sourceFilePath.length - 1)
      }
    }
    val centers = centerMap(key)
    val cltAlgo = algo match {
      case "kmeans" => new KMeans()
      case "bisecting" => new BisectingKMeans()
    }

    val jobs: Array[(Int, Array[Int])] = Utils.generateShuffleArray(sizes, rdf.rdd.getNumPartitions)

    for ((size, partitions) <- jobs) {
      var trainName = "train(size=%d)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      //var modelName = "models/logo_%s_%s_%d_%s.ml".format(algo, sourceFile, size, beginTime)
      var trainParts = (size * partitionsUnit * subs).toInt
      var predictParts = trainParts + tests
      val (trainRdd, testRdd) = rdf.rdd.getTrainAndPredictPartitions(partitions, trainParts, predictParts)
      cltAlgo.trainRdd = trainRdd
      cltAlgo.predictRdd = testRdd
      cltAlgo.trainName = trainName
      cltAlgo.centers = centers
      val inputPath = modelPath + "_" + size
      cltAlgo.run(spark, true, inputPath, true)
    }
  }
}
