package org.apache.spark.logo.ml.classification

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import java.io.Serializable

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/21 15:39
 * @desc
 */
abstract class LogoClassification extends Serializable {
  var trainRdd: RDD[Row]
  var predictRdd: RDD[Row]
  val useScore: Boolean = true
  val tail: Double = 0.05
}
