
package org.apache.spark.datagen.clusters

import breeze.linalg.DenseVector

object SequenceMapping {

  def sequenceMappingFunc(mappings: Array[Types.MappingFunc]): Types.MappingFunc = {
    v: DenseVector[Double] => {
      var r: DenseVector[Double] = v
      for (mapping <- mappings) {
        r = mapping(r)
      }
      r
    }
  }
}
