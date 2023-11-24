package org.apache.spark.logo.etl.wrappers

import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.rdd
import org.apache.spark.rdd.RDD
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.{Row, RspDataset}
import smile.data
import smile.data.DataFrame

import scala.collection.mutable

/**
 * @Author Lingxiang Zhao
 * @Date 2023/9/6 17:17
 * @desc
 */
object BasicWrappers {

  object toSmileDataFrame extends DataWrapper[smile.data.DataFrame] {
    override def apply(inputData: RDD[Row]): RspRDD[DataFrame] = {
      println("读入parquet文件，向Smile算法库 分类 数据格式兼容....")

      val compatible_Smile_Parquet_Classification: RDD[data.DataFrame] = inputData.glom().map(
        f => (
          f.map(r => r.getInt(0)),
          f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
        )
      ).map(
        f => {
          val labelFrame = data.DataFrame.of(f._1.map(l => Array(l)), "Y")
          val featureFrame = data.DataFrame.of(f._2)
          val trainFrame = featureFrame.merge(labelFrame)
          trainFrame
        }
      )
      new RspRDD(compatible_Smile_Parquet_Classification)
    }
  }

  object toMatrix extends DataWrapper[Array[Array[Int]]] {
    def apply(inputData: RDD[Row]): RspRDD[Array[Array[Int]]] = {
      println("读入parquet文件，向Smile算法库 频繁项集 数据格式兼容....")

      val compatible_Smile_Parquet_FPG: RDD[Array[Array[Int]]] = inputData
        .map(_.get(0).asInstanceOf[mutable.WrappedArray[Int]].toArray)
        .glom()

      println("--------------")
      new RspRDD(compatible_Smile_Parquet_FPG)
    }
  }

  object txtToString extends DataWrapper[Array[Array[Int]]] {
    def apply(inputData: RDD[Row]): RspRDD[Array[Array[Int]]] = {
      println("text read to string....")
      val value: RDD[String] = inputData.map((f: Row) => f.mkString(" "))
      val transaction = value.map((_: String).split(" ").map(_.toInt)).glom()
      println("--------------")
      new RspRDD(transaction)
    }
  }

  object txtToStrinWithoutGlom extends DataWrapper[Array[Int]] {
    def apply(inputData: RDD[Row]): RspRDD[Array[Int]] = {
      println("读入text文件，向Smile算法库 频繁项集 数据格式兼容....")
      val value: RDD[String] = inputData.map((f: Row) => f.mkString(" "))
      val transaction = value.map((_: String).split(" ").map(_.toInt))
      println("--------------")
      new RspRDD(transaction)
    }
  }


  object toMatrixRDD extends DataWrapper[(Array[Int], Array[Array[Double]])] {
    def apply(inputData: RDD[Row]): RspRDD[(Array[Int], Array[Array[Double]])] = {
      val rdd: RDD[(Array[Int], Array[Array[Double]])] = inputData.glom().map(
        f => (
          f.map(r => r.getInt(0)),
          f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
        )
      )
      new RspRDD(rdd)
    }
  }

  /**
   * 不带标签
   */
  object toMatrixRDDWithoutLabel extends DataWrapper[Array[Array[Double]]] {
    def apply(inputData: RDD[Row]): RspRDD[Array[Array[Double]]] = {
      val rdd: RDD[Array[Array[Double]]] = inputData.glom().map(
        f =>
          f.map(r => r.get(1).asInstanceOf[DenseVector].toArray)
      )
      new RspRDD(rdd)
    }
  }

}
