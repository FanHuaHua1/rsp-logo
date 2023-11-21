package org.apache.spark.logo.ml.classification

import org.apache.spark.logo.Utils
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.collection.mutable.{WrappedArray, Map => MutablMap}

object Entrypoint {

  /**
   *
   * @param spark
   * @param args: Array[String], spark|logo commands...
   */
  def onArgs(spark: SparkSession, args: Array[String]): Unit = {
    args(0) match {
      case "logoShuffle" => fitLogoShuffle(spark, args, true)
      case _ => printf("Unknown type: %s\n", args(0))
    }
  }


  /**
   * 带模型地址
   * @param spark
   * @param args
   * @param useScore
   */
  def fitLogoShuffle(spark: SparkSession, args: Array[String], useScore: Boolean): Unit = {
    logoShuffle(
      spark, args(1), args(2), args(3),
      args(4).toDouble, args(5).toDouble, args(6).toDouble,
      args(7).toDouble, args.slice(8, args.length).map(_.toInt),
      useScore
    )
  }
  /**
   *
   * @param spark
   * @param algo: String, 算法 DT|LR|RF|SVM
   * @param sourceFile: String, 数据文件
   * @param subs: Double, rsp取块比例
   * @param predicts: Double, 预测集块数
   * @param tails: Double, 头尾筛选比例
   * @param partitionsUnit: Double, 单位数据块数, 计算取块数 = (size * partitionsUnit).toInt()
   * @param sizes: Array[Int], 训练数据集大小
   * @param useScore
   */
  def logoShuffle(spark: SparkSession,
                  algo: String,
                  sourceFile: String,
                  modelPath:String,
                  subs: Double,
                  predicts: Double,
                  tails: Double,
                  partitionsUnit: Double,
                  sizes: Array[Int], useScore: Boolean=true): Unit = {
    println ("algo: " + algo)
    println ("sourceFile: " + sourceFile)
    println ("modelPath: " + modelPath)
    println ("subs: " + subs)
    println ("predicts: " + predicts)
    println ("tails: " + tails)
    println ("partitionsUnit: " + partitionsUnit)
    println ("sizes: " + sizes.toList)

    var rdf = spark.rspRead.parquet(sourceFile)
    val jobs: Array[(Int, Array[Int])] = Utils.generateShuffleArray(sizes, rdf.rdd.getNumPartitions)
    //println("jobs: " + jobs.map(_._1).toList + jobs.map(_._2.toList).toList)
    var predictBlocks = math.ceil(predicts).toInt
    var clfAlgo = algo match {
      case "LR" => new LogisticRegression()
      case "DT" => new DecisionTree()
      case "RF" => new RandomForest()
    }

    for ((size, partitions) <- jobs) {
      var trainName = "train(size=%d)".format(size)
      //var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      var trainParts = (size * partitionsUnit * subs).toInt
      var predictParts = trainParts + predictBlocks
      printf(
        "size=%d, trainParts(%d)=%s, predictParts(%d)=%s\n",
        size,
        trainParts, partitions.slice(0, trainParts).toList,
        predictParts-trainParts, partitions.slice(trainParts, predictParts).toList
      )
      printf("%s\n", trainName)
      val (trainRdd, predictRdd) = rdf.rdd.getTrainAndPredictPartitions(partitions, trainParts, predictParts)
      clfAlgo.trainRdd = trainRdd
      clfAlgo.predictRdd = predictRdd
      val inputPath = modelPath + "_" + size
      clfAlgo.run(saveModel = true, inputPath, doEvaluate = false)
    }
  }


  def votePrediction(param: (Long, (Iterable[(Long, Array[Int])], Int))): (Long, Array[Int]) = {
    val labels = param._2._1.map(_._2).toArray
    val result = new Array[Int](param._2._2)
    val members = labels.length
    val counts = MutablMap[Int, Int]()
    for (i <- 0 until param._2._2) {
      for (m <- 0 until members) {
        counts(labels(m)(i)) = counts.getOrElse(labels(m)(i), 0) + 1
      }
      result(i) = counts.maxBy(_._2)._1
      counts.clear()
    }

    (param._1, result)
  }

}
