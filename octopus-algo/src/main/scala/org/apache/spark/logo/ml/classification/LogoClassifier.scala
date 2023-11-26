package org.apache.spark.logo.ml.classification

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, RspDataset}

import scala.collection.mutable.{Map => MutablMap}

/**
 *
 * @tparam M: TrainModel class
 */
trait LogoClassifier[M]{
  /**
   * 标签数据类型
   */
  type LABEL = Array[Int]
  /**
   * 特征数据类型
   */
  type FEATURE = Array[Array[Double]]

  /**
   *
   * @param rdd: 原始数据RDD
   * @return trainData: RDD[ (Array[Int], Array[ Array[Double] ]) ]
   */
  def etl(rdd: RDD[Row]): RDD[(LABEL, FEATURE)]

  /**
   *
   * @param sample: 用于训练的数据类型
   * @return (model: M，trainTimeSeconds: Double)
   */
  def LO(sample: (LABEL, FEATURE)): (M, Double)
  def GO(modelRdd: RDD[(M, Double, Double)]): RDD[(M, Double)]

  /**
   *
   * @param prediction: 模型预测值
   * @param label: 原始标签
   * @return accuracy: Double
   */
  def estimator(prediction: LABEL, label: LABEL): Double

  /**
   *
   * @param model: M, 模型
   * @param features: FEATURE, 预测数据集
   * @return prediction: LABEL，预测结果
   */
  def predictor(model: M, features: FEATURE): LABEL


  def run(isCrossDomain:Boolean, saveModel:Boolean, saveModelPath:String, doEvaluate: Boolean): RDD[(M, Double)]


  def getValuedModels(modelRdd: RDD[(M, Double, Double)], useScore: Boolean, tail: Double = 0.05): RDD[(M, Double)] = {
    if (useScore) {
      val factors = modelRdd.map (_._3).collect().sorted
      val mcount = factors.length
      printf ("Model count: %d\n", mcount)
      var tcount = (mcount * tail).toInt
      if (tcount < 1) {
        tcount = 1
      }
      printf ("Tail count: %d\n", tcount)
      val (minAcc, maxAcc) = (factors (tcount), factors (mcount - tcount - 1) )
      printf ("Score range: (%f, %f)\n", minAcc, maxAcc)
      modelRdd.filter (item => minAcc <= item._3 && item._3 <= maxAcc).map (item => (item._1, item._3) )
    } else {
      modelRdd.map (item => (item._1, 1.0) )
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


  def evaluate(modelRdd: RDD[(M, Double)], predictRdd: RDD[Row]): Unit = {
    if (predictRdd == null) {
      printf("Accuracy: %f\n", 1.0)
      return
    }
    printf("ValuedModel count: %d\n", modelRdd.count())
    val predictWithIndex = etl(predictRdd).zipWithIndex()
    val predicts = predictWithIndex.map(item => (item._2, item._1, item._1._1.length))
    //val value: RDD[((M, Double), (Long, (LABEL, FEATURE), Int))] = modelRdd.cartesian(predicts)
    val prediction = modelRdd.cartesian(predicts).map(
      item => (item._2._1, predictor(item._1._1, item._2._2._2))
    ).groupBy(_._1)
    val sizeRDD = predicts.map(item => (item._1, item._3))
    val rspPredict = prediction.join(sizeRDD).map(votePrediction)
    val indexedLabels = predictWithIndex.map(item => (item._2, item._1._1))
    val rspAcc = rspPredict.join(indexedLabels).map(
      item => (estimator(item._2._1, item._2._2), item._2._1.length)
    )
    val acc = rspAcc.map(item => item._1 * item._2).sum / rspAcc.map(_._2).sum
    printf("Accuracy: %f\n", acc)
  }
}
