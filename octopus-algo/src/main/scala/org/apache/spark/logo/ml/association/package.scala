package org.apache.spark.logo.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.Row
import smile.association.ItemSet

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/15 11:09
 * @desc
 */
package object association {
  def fpgVote(minSupport: Double, itemsets: RDD[Row], isCrossDomain:Boolean = false, saveModel: Boolean = false, saveModelPath: String = null) = {
    VoteFPGrowth(minSupport, itemsets, isCrossDomain, saveModel, saveModelPath)
  }

  def fpgBroadcast(minSupport: Double, itemsets: RDD[Row]) = {
    BroadcastFPGrowth(minSupport, itemsets)
  }
}
