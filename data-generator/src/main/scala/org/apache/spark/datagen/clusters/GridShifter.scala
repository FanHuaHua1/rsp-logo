
package org.apache.spark.datagen.clusters

import breeze.linalg.DenseVector
import breeze.stats.distributions.Uniform


object GridShifter {

  class Cartesian(arrays: Array[Double]*) extends Iterator[Array[Double]] {

    var l = arrays.size
    var sizes = arrays.map(_.length).toArray
    var points = arrays.map(_ => 0).toArray
    var index = Array.range(0, l)
    var x: Int = l - 1

    override def hasNext: Boolean = {
      return x >= 0
    }

    override def next(): Array[Double] = {
      var r = index.map(i => arrays(i)(points(i)))
      movePoints()
      return r
    }


    def movePoints(): Unit = {
      while (x >= 0) {
        if (points(x) + 1 < sizes(x)) {
          points(x) = points(x) + 1
          if (x < l-1) {
            x = l-1
          }
          return
        } else {
          points(x) = 0
          x = x - 1
        }
      }
    }

  }

  def cartesian(arrays: Array[Double]*): Iterator[Array[Double]] = {
    return new Cartesian(arrays: _*)

  }

  def randomVectorsByBlock(n: Int, dimension: Int): Array[Array[Double]] = {
    var parts = math.ceil(math.pow(n, 1.0 / dimension)).toInt
    var arrays = Array.range(0, dimension).map(_ => Array.range(0, parts).map(_.toDouble/parts))
    var c = cartesian(arrays: _*)

    var dist = Uniform(0, 1)
    var grids = c.slice(0, n).toArray
    grids.map(_.map(_ + dist.draw()/parts))

  }

  def grid(mapping: Types.MappingFunc = null): Types.ClusterShiftFunc = {
    if (mapping==null) {
      return (n: Int, dimension: Int) => {
        randomVectorsByBlock(n, dimension).map(arr => DenseVector(arr))
      }
    } else {
      return (n: Int, dimension: Int) => {
        randomVectorsByBlock(n, dimension).map(arr => DenseVector(arr)).map(mapping)
      }
    }

  }

  class GridEstimation() {

  }



}
