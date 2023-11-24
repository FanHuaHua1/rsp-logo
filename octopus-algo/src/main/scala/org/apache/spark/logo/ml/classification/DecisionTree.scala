package org.apache.spark.logo.ml.classification

import org.apache.spark.logo.Utils
import org.apache.spark.logo.etl.wrappers.BasicWrappers
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.Row
import smile.classification.{cart, DecisionTree => smileDT}
import smile.data.DataFrame
import smile.data.formula.Formula
import smile.validation.metric.Accuracy

class DecisionTree(
        var trainRdd: RDD[Row],
        var predictRdd: RDD[Row],
        useScore: Boolean = true,
        tail: Double = 0.05,
        var maxDepth: Int = 10,
        var nodeSize: Int = 10) extends ClassificationJob[smileDT]{

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
  override def predictor(model: smileDT, features: FEATURE): LABEL = {
    val testFeatures: Array[Array[Double]] = features
    model.predict(DataFrame.of(testFeatures))
  }


  /**
   *
   * @param sample  : 用于训练的数据类型
   * @return (model: M，trainTimeSeconds: Double)
   */
  override def LO(sample: (LABEL, FEATURE)): (smileDT, Double) = {
    val trainLabel: Array[Array[Int]] = sample._1.map(l => Array(l))
    val trainFeatures: Array[Array[Double]] = sample._2
    val featureFrame: DataFrame = DataFrame.of(trainFeatures) //特征列
    val labelFrame: DataFrame = DataFrame.of(trainLabel, "Y") //标签列
    val formula: Formula = Formula.lhs("Y") //创建Formula，设定除Y之外都是特征
    val startTime = System.nanoTime
    val trainFrame = featureFrame.merge(labelFrame)
    val tree: smileDT = cart(formula, trainFrame, maxDepth = this.maxDepth, nodeSize = this.nodeSize)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒
    (tree, duration)
  }


  override def GO(modelRdd: RDD[(smileDT, Double, Double)]): RDD[(smileDT, Double)] = {
    getValuedModels(modelRdd, useScore, tail)
  }

}

object DecisionTree {
  def apply(rspRdd: RspRDD[Row], trainParNums: Int, predictParNums: Int, useScore: Boolean = true, tail: Double= 0.05, maxDepth: Int = 10, nodeSize: Int = 10): RDD[(smileDT, Double)] = {
    val (trainRdd, predictRdd) = Utils.transform(rspRdd, trainParNums, predictParNums)
    new DecisionTree(trainRdd, predictRdd, useScore, tail, maxDepth, nodeSize).run(false, saveModel = false, "", doEvaluate = false)
  }
}
