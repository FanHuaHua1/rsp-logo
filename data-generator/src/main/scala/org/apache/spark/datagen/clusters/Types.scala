
package org.apache.spark.datagen.clusters

import breeze.linalg.DenseVector

import org.apache.spark.rdd.RDD

object Types {

  type DistributionRDDFunc = (Int) => RDD[Double]

  type DistributionConstructor = () => breeze.stats.distributions.Rand[Double]

  type MappingFunc = (DenseVector[Double]) => DenseVector[Double]

  /** Type of function that generate shift vectors for cluster vectors.
   *
   * @tparam n: Int, numbers of shift vectors to generate.
   * @tparam dimension: Int, dimension of shift vectors.
   * @return shift vectors.
   */
  type ClusterShiftFunc = (Int, Int) => Array[DenseVector[Double]]


}
