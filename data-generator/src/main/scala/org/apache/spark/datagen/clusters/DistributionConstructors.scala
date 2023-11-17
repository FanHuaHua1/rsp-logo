
package org.apache.spark.datagen.clusters

import breeze.stats.distributions._

object DistributionConstructors {

  class ConstantGen(value: Double) extends Rand[Double] {
    override def draw(): Double = {
      value
    }
  }

  def constant(value: Double): Types.DistributionConstructor = {
    () => {
      new ConstantGen(value)
    }
  }

  def uniform(low: Double, high: Double): Types.DistributionConstructor = {
    () => {
      Uniform(low, high)
    }
  }

  def gaussian(mu: Double, sigma: Double): Types.DistributionConstructor = {
    () => {
      Gaussian(mu, sigma)
    }

  }

  def gamma(shape: Double, scale: Double): Types.DistributionConstructor = {
    () => {
      Gamma(shape, scale)
    }
  }


}
