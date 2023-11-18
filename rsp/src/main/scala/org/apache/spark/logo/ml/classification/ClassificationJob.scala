package org.apache.spark.logo.ml.classification

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import java.io.Serializable

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/21 15:39
 * @desc
 */
abstract class ClassificationJob[M] extends Serializable with LogoClassifier[M]{
  var trainRdd: RDD[Row]
  var predictRdd: RDD[Row]
  val useScore: Boolean = true
  val tail: Double = 0.05

  override def run(saveModel: Boolean, saveModelPath: String, doEvaluate: Boolean): RDD[(M, Double)] = {
    val modelRdd: RDD[(M, Double, Double)] = etl(trainRdd).map(
      sample => {
        val trainSize = (sample._1.length * 0.9).toInt
        val trainSample = (sample._1.slice(0, trainSize), sample._2.slice(0, trainSize))
        val testSample = (sample._1.slice(trainSize, sample._1.length), sample._2.slice(trainSize, sample._1.length))
        val (model, duration) = LO(trainSample)
        val accuracy = estimator(predictor(model, testSample._2), testSample._1)
        (model, duration, accuracy)
      }
    )
    val valuedModels: RDD[(M, Double)] = GO(modelRdd)
    if (saveModel) {
      val saveModelPair: RDD[Pair[M, Double]] = valuedModels.map(f => new Pair(f._1, f._2))
      saveModelPair.saveAsObjectFile(saveModelPath)
    }
    if (doEvaluate) {
      evaluate(valuedModels, predictRdd)
    }
    valuedModels
  }
}
