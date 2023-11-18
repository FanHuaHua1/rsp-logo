package org.apache.spark.logo.ml

import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.{Row}
import smile.classification.{RandomForest => smileRF, DecisionTree => smileDT, LogisticRegression => smileLR}


package object classification {

//  def logoDecisionTree(formula: Formula, data: DataFrame, splitRule: SplitRule = SplitRule.GINI, maxDepth: Int = 20, maxNodes: Int = 0, nodeSize: Int = 5): smile.classification.DecisionTree = time("LOGO Decision Tree"){
//    cart(formula, data, splitRule, maxDepth, maxNodes, nodeSize)
//  }
//
//  def logoRandomForest(formula: Formula, data: DataFrame, ntrees: Int = 500, mtry: Int = 0,
//                       splitRule: SplitRule = SplitRule.GINI, maxDepth: Int = 20, maxNodes: Int = 500,
//                       nodeSize: Int = 1, subsample: Double = 1.0, classWeight: Array[Int] = null,
//                       seeds: LongStream = null): RandomForest = time("LOGO Random Forest") {
//
//    randomForest(formula,data: DataFrame,ntrees,mtry,splitRule, maxDepth, maxNodes, nodeSize, subsample, classWeight, seeds)
//  }

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
