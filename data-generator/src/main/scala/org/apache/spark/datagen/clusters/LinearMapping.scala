
package org.apache.spark.datagen.clusters

import breeze.linalg.{DenseMatrix, DenseVector}


object LinearMapping {

  class LinearMappingConstructor(dimension: Int) extends Serializable {
    var matrix = DenseMatrix.eye[Double](dimension)

    def rotation2d(theta: Double): LinearMappingConstructor = {
      assert(dimension==2)
      matrix = LinearMapping.rotation2d(theta) * matrix
      this
    }

    def rodrigues(axis: DenseVector[Double], theta: Double): LinearMappingConstructor = {
      assert(dimension==3)
      matrix = LinearMapping.rodrigues(axis, theta) * matrix
      this
    }

    def dot(temp: DenseMatrix[Double]): LinearMappingConstructor = {
      assert(dimension == temp.rows && dimension == temp.cols)
      matrix = temp * matrix
      this
    }

    def mappingFunc(): Types.MappingFunc = {
      return LinearMapping.matrixMappingFunc(matrix)
    }

  }

  def constructor(dimension: Int): LinearMappingConstructor = {
    return new LinearMappingConstructor(dimension)
  }


  def matrixMappingFunc(m: DenseMatrix[Double]): Types.MappingFunc = {
    v: DenseVector[Double] => m * v
  }

  def rotation2d(theta: Double): DenseMatrix[Double] = {
    var sinT = math.sin(theta)
    var cosT = math.cos(theta)
    DenseMatrix((cosT, -sinT), (sinT, cosT))
  }

  def rotation2dMapping(theta: Double) : Types.MappingFunc = {
    matrixMappingFunc(rotation2d(theta))
  }

  def rodrigues(axis: DenseVector[Double], theta: Double): DenseMatrix[Double] = {
    assert(axis.size == 3)
    var normalized = breeze.linalg.normalize(axis)
    var sinT = math.sin(theta)
    var cosT = math.cos(theta)
    DenseMatrix.eye[Double](3) * cosT +
      (normalized * normalized.t) * (1 - cosT) +
      DenseMatrix[(Double, Double, Double), Double](
        (0, -normalized(2), normalized(1)),
        (normalized(2), 0, -normalized(0)),
        (-normalized(1), normalized(0), 0)
      ) * sinT
  }

  def rodriguesMapping(axis: DenseVector[Double], theta: Double): Types.MappingFunc = {
    matrixMappingFunc(rodrigues(axis, theta))
  }

}
