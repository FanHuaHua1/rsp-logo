package org.apache.spark.logo.ml.clustering

import org.apache.spark.logo.Utils
import org.apache.spark.logo.etl.wrappers.BasicWrappers
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.Row
import org.apache.spark.logo.ml.clustering.bmutils.LOGOBisectingKMeans

class BisectingKMeans(
              var trainRdd: RDD[Row],
              var predictRdd: RDD[Row],
              var maxIterations: Int = 2,
              var tol: Double = 1.0E-4) extends ClusteringJob[LOGOBisectingKMeans]{

  def this() = this(null, null)

  def etl(rdd: RDD[Row]): RDD[(Array[Int], Array[Array[Double]])] = BasicWrappers.toMatrixRDD(rdd)


  def getPurity(model: LOGOBisectingKMeans, test: (LABEL, FEATURE)): Double = {
    val predictions = test._2.map(model.predict)
    var p = predictions.zip(test._1)
      .count(r => r._1 == r._2)
      .toDouble / predictions.length
    if (p < 0.5) {
      p = 1 - p
    }
    p
  }

  /**
   *
   * @param sample  : 用于训练的数据类型
   * @return (model: M，trainTimeSeconds: Double)
   */
  override def LO(features:FEATURE): (LOGOBisectingKMeans, Double) = {
    val bisecting = new LOGOBisectingKMeans(k, maxIterations);
    val startTime = System.nanoTime
    bisecting.clustering(features)
    val duration = (System.nanoTime - startTime) * 0.000000001 //System.nanoTime为纳秒，转化为秒

    (bisecting, duration)
  }

  def getCentroids(model: LOGOBisectingKMeans): Array[Array[Double]] = {
    model.centroids
  }

  override def GO(modelsRdd: RDD[(LOGOBisectingKMeans, Double)]): RDD[(LOGOBisectingKMeans, Double)] = {
    var resampledRDD = modelsRdd.map(item => getCentroids(item._1)).flatMap(arr=>arr).coalesce(1).glom()
    var modelRDD = resampledRDD.map(LO)
    modelRDD
    //var centers = modelRDD.map(r => getCentroids(r._1)).collect()(0)
  }
}

object BisectingKMeans {
  def apply(rspRdd: RspRDD[Row], trainParNums: Int, predictParNums: Int, centerPath: String = "", maxIterations: Int = 100, tol: Double = 1.0E-4): Array[Array[Double]] = {
    val (trainRdd, predictRdd) = Utils.transform(rspRdd, trainParNums, predictParNums)
    new BisectingKMeans(trainRdd, predictRdd).run("")
  }

}
