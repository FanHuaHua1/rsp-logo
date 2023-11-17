
package org.apache.spark.datagen.clusters

import breeze.linalg.DenseVector

case class SingleCluster(constructors: Array[Types.DistributionConstructor],
                    mapping: Types.MappingFunc = null) extends Cluster with Serializable {


  override def dimension(): Int = {
    return constructors.size
  }

  override def generate(count: Int): Array[DenseVector[Double]] = {
    return generator(count).toArray
  }

  override def generator(count: Int): Iterator[DenseVector[Double]] = {
    var distributions = constructors.map(_())
    var iterator = Iterator.range(0, count).map(
      _ => DenseVector(distributions.map(_.draw()))
    )

    if (mapping == null) {
      return iterator
    } else {
      return iterator.map(mapping)
    }
  }
}