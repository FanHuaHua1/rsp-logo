package org.apache.spark.logo.etl.wrappers

import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.{Row, RspDataset}


trait DataWrapper[T] {
  def apply(inputData: RDD[Row]): RspRDD[T]
}
