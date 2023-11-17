
package org.apache.spark.datagen.clusters

import breeze.linalg.DenseVector

object Shifters {

  def zeros(): Types.ClusterShiftFunc = {
    (numbers: Int, dimension: Int) => Array.range(
      0, numbers
    ).map(
        _ => DenseVector.zeros[Double](dimension)
    )
  }
}
