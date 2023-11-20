package org.apache.spark.logo.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.{Row}
import smile.classification.{RandomForest => smileRF, DecisionTree => smileDT, LogisticRegression => smileLR}


package object classification {
  def lr(rspRdd: RspRDD[Row], trainParNums: Int, predictParNums: Int, useScore: Boolean = true, tail: Double= 0.05, maxIter: Int, regParam: Double) : RDD[(smileLR, Double)] = {
    LogisticRegression(rspRdd, trainParNums, predictParNums, useScore, tail , maxIter, regParam)
  }

  def dt(rspRdd: RspRDD[Row], trainParNums: Int, predictParNums: Int, useScore: Boolean = true, tail: Double= 0.05, maxDepth: Int = 10, nodeSize: Int = 10): RDD[(smileDT, Double)] = {
    DecisionTree(rspRdd, trainParNums, predictParNums, useScore, tail, maxDepth, nodeSize)
  }

  def rf(rspRdd: RspRDD[Row], trainParNums: Int, predictParNums: Int, useScore: Boolean = true, tail: Double = 0.05, nTrees: Int = 20, maxDepth: Int = 5): RDD[(smileRF, Double)] = {
    RandomForest(rspRdd, trainParNums, predictParNums, useScore, tail, nTrees, maxDepth)
  }
}
