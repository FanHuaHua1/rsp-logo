
package org.apache.spark.datagen.clusters

//import breeze.linalg.DenseVector

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.DenseVector

case class ClusterGenerator(
                        dimension: Int,
                        clusters: Array[Cluster],
                        shifter: Types.ClusterShiftFunc
                      ) extends Serializable {

  for (cluster <- clusters) {
    assert(cluster.dimension() == dimension)
  }

  def generate(spark: SparkSession,
               partitions: Int,
               counts: Array[Int]): DataFrame = {
    import spark.implicits._
    var number = counts.length
    assert(counts.length == clusters.length)
    var indexRDD = spark.sparkContext.parallelize(Seq[Int](), partitions)

    var shiftVectors = shifter(counts.length, dimension)

    indexRDD.mapPartitions(
      _ => Array.range(0, number).map(
        i => clusters(i).generator(counts(i)).map(
          v => (i, new DenseVector((v + shiftVectors(i)).toArray))
        )
      ).reduce(_ ++ _)
    ).toDF("label", "features")
  }

  def generate(counts: Array[Int]) : (Array[Int], Array[Array[Double]]) = {
    var shiftVectors = shifter(counts.length, dimension)
    var number = counts.length
    assert(counts.length == clusters.length)
    var features = Array.range(0, number).map(
      i => clusters(i).generator(counts(i)).map(
        v => (v + shiftVectors(i)).toArray
      )
    ).reduce(_ ++ _).toArray
    var label = Array.range(0, number).map(i => Array.fill(counts(i)) {i}).reduce(_ ++ _)
    (label, features)
  }

}
