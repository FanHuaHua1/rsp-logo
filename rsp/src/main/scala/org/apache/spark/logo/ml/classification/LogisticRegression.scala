package org.apache.spark.logo.ml.classification

import org.apache.spark.api.java.JavaRDD.fromRDD
import org.apache.spark.logo.Utils
import org.apache.spark.logo.etl.wrappers.BasicWrappers
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import javafx.util.Pair
import org.apache.spark.sql.{Row, RspDataset}
import smile.classification.{LogisticRegression => smileLR}
import smile.validation.metric.Accuracy

import java.io.Serializable

class LogisticRegression(
                          var trainRdd: RDD[Row],
                          var predictRdd: RDD[Row],
                          useScore: Boolean = true,
                          tail: Double = 0.05,
                          var maxIter: Int = 500,
                          var regParam: Double = 0.1) extends LogoClassification with LogoClassifier[smileLR]{

  def this() = this(null, null)

  def etl(rdd: RDD[Row]): RDD[(Array[Int], Array[Array[Double]])] = BasicWrappers.toMatrixRDD(rdd)

  /**
   *
   * @param prediction : 模型预测值
   * @param label      : 原始标签
   * @return accuracy: Double
   */
  override def estimator(prediction: LABEL, label: LABEL): Double = {
    Accuracy.of(label, prediction)
  }

  /**
   *
   * @param model    : M, 模型
   * @param features : FEATURE, 预测数据集
   * @return prediction: LABEL，预测结果
   */
  override def predictor(model: smileLR, features: FEATURE): LABEL = {
    val testFeatures: Array[Array[Double]] = features
    model.predict(testFeatures)
  }


  /**
   *
   * @param sample  : 用于训练的数据类型
   * @return (model: M，trainTimeSeconds: Double)
   */
  override def LO(sample: (LABEL, FEATURE)): (smileLR, Double) = {
    val trainLabel: Array[Int] = sample._1
    val trainFeatures: Array[Array[Double]] = sample._2
    val startTime = System.nanoTime
    val model = smileLR.fit(trainFeatures, trainLabel)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    (model, duration)
  }


  override def GO(modelRdd: RDD[(smileLR, Double, Double)]): RDD[(smileLR, Double)] = {
    getValuedModels(modelRdd, useScore, tail)
  }

  override def run(saveModel:Boolean, saveModelPath:String, doEvaluate: Boolean): RDD[(smileLR, Double)]  = {
    val modelRdd: RDD[(smileLR, Double, Double)] = etl(trainRdd).map(
      sample => {
        val trainSize = (sample._1.length * 0.9).toInt
        val trainSample = (sample._1.slice(0, trainSize), sample._2.slice(0, trainSize))
        val testSample = (sample._1.slice(trainSize, sample._1.length), sample._2.slice(trainSize, sample._1.length))
        val (model, duration) = LO(trainSample)
        val accuracy = estimator(predictor(model, testSample._2), testSample._1)
        (model, duration, accuracy)
      }
    )
    val valuedModels: RDD[(smileLR, Double)] = GO(modelRdd)
    if(saveModel){
      val saveModelPair: RDD[Pair[smileLR, Double]] = valuedModels.map(f => new Pair(f._1, f._2))
      saveModelPair.saveAsObjectFile(saveModelPath)
    }
    if(doEvaluate){
      evaluate(valuedModels, predictRdd)
    }
    valuedModels
  }

}

object LogisticRegression {
  def apply(rspRdd: RspRDD[Row], trainParNums: Int, predictParNums: Int, useScore: Boolean = true, tail: Double= 0.05, maxIter: Int = 500, regParam: Double = 0.1): RDD[(smileLR, Double)] = {
    val (trainRdd, predictRdd) = Utils.transform(rspRdd, trainParNums, predictParNums)
    new LogisticRegression(trainRdd, predictRdd, useScore, tail, maxIter, regParam).run(saveModel = false, "", doEvaluate = false)
  }

}