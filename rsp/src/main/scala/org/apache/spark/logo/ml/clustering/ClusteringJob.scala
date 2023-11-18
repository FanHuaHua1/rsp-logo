package org.apache.spark.logo.ml.clustering

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.numericRDDToDoubleRDDFunctions
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.RspContext._
import spire.ClassTag
import java.io.Serializable
import scala.reflect.ClassTag

/**
 * @Author Lingxiang Zhao
 * @Date 2023/11/17 20:39
 * @desc
 */
abstract class ClusteringJob[M] extends Serializable with LogoClustering[M] {
  var trainRdd: RDD[Row]
  var predictRdd: RDD[Row]
  var k: Int = 2
  var centers: Array[Array[Double]] = null
  val trainName: String = ""

  def run(modelName: String = ""): FEATURE = {
    printf("%s start\n", trainName)
    val begin = System.nanoTime()
    val rdd = etl(trainRdd)
    val modelsRDD = rdd.map(r => {
      var (m, d) = LO(r._2)
      (getCentroids(m), d)
    })
    val trainEnd = System.nanoTime()
    printf("Train time: %f\n", (trainEnd - begin) * 1e-9)

    val flatRDD = modelsRDD
      .flatMap(r => r._1)
    val groupedCenters = flatRDD.collect()

    val flatEnd = System.nanoTime()
    printf("Flat time: %f\n", (flatEnd - trainEnd) * 1e-9)
    printf("Flat count: %d\n", groupedCenters.length)

    var (model, _) = LO(groupedCenters)
    val predictCenters = getCentroids(model)
    val duration = System.nanoTime() - begin
    printf("%s finished\n", trainName)
    printf("Time spend: %f\n", duration * 1e-9)
//
//    val distance = minDistance(centers, predictCenters)
//    printf("%s Distance: %f\n", trainName, distance)
//    if (predictRdd == null) {
//      printf("%s Purity: %f\n", trainName, 1.0)
//      return predictCenters
//    }
//
//    var purity = etl(predictRdd).map(item => getPurity(model, item)).mean()
//    printf("%s Purity: %f\n", trainName, purity)
//
//    centers.foreach(showArray)
//    predictCenters.foreach(showArray)
    //    model.write.save(modelName)
    predictCenters
  }


  def minDistance(real: Array[Array[Double]], predict: Array[Array[Double]]): Double = {
    real.map(
      rc => predict.map(pc => pc.zip(rc).map(item => math.pow(item._1 - item._2, 2)).sum).min
    ).sum
  }

  def showArray(vector: Array[Double]): Unit = {
    println((new DenseVector(vector)).toString)
  }

}
