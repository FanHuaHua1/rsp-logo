
package org.apache.spark.datagen.clusters

import breeze.linalg.DenseVector

trait Cluster {
  def generator(count: Int): Iterator[DenseVector[Double]]
  def generate(count: Int): Array[DenseVector[Double]]
  def dimension(): Int
}
