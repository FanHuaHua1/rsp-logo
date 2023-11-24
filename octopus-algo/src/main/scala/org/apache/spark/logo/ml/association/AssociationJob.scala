package org.apache.spark.logo.ml.association

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row

import java.io.Serializable

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/21 15:39
 * @desc
 */
abstract class AssociationJob extends Serializable{
  var minSupport: Double
  var itemsets: RDD[Row]
  def runText(isCrossDomain:Boolean, saveModel: Boolean, saveModelPath: String)
}
