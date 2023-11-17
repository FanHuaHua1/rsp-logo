
package org.apache.spark.datagen.clusters

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.linalg.norm
import breeze.stats.distributions.Rand

object Hyperplane {

  def mappingFunc(vector: DenseVector[Double], distance: Double): Types.MappingFunc = {
    if (vector(vector.length-1) == 0) {
      throw new IllegalArgumentException("require: vector[-1] != 0 ")
    }

//    var factors = vector.slice(0, vector.size-1)
    var div = 1 / vector(vector.length-1)
    var factors = vector.size-1

    (v: DenseVector[Double]) => {
      var r = v
      var result = 0.0
      for (i <- 0 to factors) {
        result = result + v(i) * vector(i)
      }
      r(r.size-1) = (distance - result) * div
      r
    }
  }

  class VectorNoise(distribution: Types.DistributionConstructor,
                    vector: DenseVector[Double]) extends Serializable {
    var normFactor = norm(vector)
    var normVector = vector.map(_/normFactor)

    lazy val dist: Rand[Double] = distribution()

    def mapping(target: DenseVector[Double]): DenseVector[Double] = {
      var l = dist.draw()
      normVector.map(_*l) + target
    }

  }

  def noiseFunc(distribution: Types.DistributionConstructor,
                vector: DenseVector[Double]): Types.MappingFunc = {
    var vn = new VectorNoise(distribution, vector)
    (vector: DenseVector[Double]) => {
      vn.mapping(vector)
    }
  }

  def getHyperplaneCluster(constructors: Array[Types.DistributionConstructor],
                 vector: DenseVector[Double],
                 distance: Double, noise: Types.DistributionConstructor): Cluster = {

    SingleCluster(
      constructors,
      SequenceMapping.sequenceMappingFunc(
        Array(mappingFunc(vector, distance), noiseFunc(noise, vector))
      )
    )
  }

}
