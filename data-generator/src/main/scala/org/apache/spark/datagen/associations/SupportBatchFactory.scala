
package org.apache.spark.datagen.associations

import scala.collection.mutable.ListBuffer
import scala.util.Random


case class SupportBatchFactory(
                              items: Array[Int] = Array.range(0, 100),
                              supportRange: (Double, Double) = (0.1, 0.2),
                              unionLimits: (Double, Double) = (0.2, 0.3),
                              isolatedSupportLimit: Double = 0.1
                              ) extends Serializable {


  def randomRange(r: (Double, Double)): Double = Random.nextDouble() * (r._2 - r._1) + r._1

  def generateSupportBatch(_items: Array[Int]): Types.SupportBatch = {
    val buffer = new ListBuffer[BatchItem]()
    val sr = randomRange(supportRange)
    val others = randomRange(unionLimits) - sr
    buffer.append(new BatchItem(_items, sr))
    var size = _items.length - 1
    while (size > 0) {
      for (arr <- _items.combinations(size)) {
        buffer.append(new BatchItem(arr, Random.nextDouble()))
      }

      size -= 1
    }

    val result = buffer.toArray

    val total = result.map(_.support).slice(1, result.length).sum

    for (i <- 1 until result.length) {
      result(i).support = result(i).support * others / total
    }

    result.map(_.toTuple())
  }

  def generateSupportBatches(sizes: Array[Int]): Array[Types.SupportBatch] = {
    var index = 0
    val results = new ListBuffer[Types.SupportBatch]()

    for (size <- sizes ) {
      results.append(
        generateSupportBatch(items.slice(index, index + size))
      )

      index += size
    }

    for (item <- items.slice(index, items.length)) {
      results.append(
        Array((Array(item), randomRange((0, isolatedSupportLimit))))
      )
    }

    results.toArray
  }

}
