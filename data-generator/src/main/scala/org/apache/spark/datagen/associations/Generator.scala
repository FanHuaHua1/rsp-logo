
package org.apache.spark.datagen.associations

import scala.collection.mutable.ListBuffer
import scala.util.Random

object Generator extends Serializable {

  implicit class ArrayToStringHelper[T](array: Array[T]) {

    def toStr: String = {
      if (array.length > 0) {

      "[" + array.map(_.toString).reduce(_ + ", " + _) + "]"
      } else {
        "[]"
      }
    }
  }

  def calculateCount(batches: Array[Types.SupportBatch], size: Long): Int = {
    (size / batches.map(_.map(item => item._1.length * item._2).sum).sum / 4).toInt
  }

  def generateFromBatch(batches: Array[Types.SupportBatch], count: Int): Array[ListBuffer[Int]] = {
    val temp = Iterator.range(0, count).map(r => new ListBuffer[Int]()).toArray

    var index = 0
    val idxParent = List.range(0, count)
    var indexes: Array[Int] = null

    for (batch <- batches) {
      indexes = Random.shuffle(idxParent).toArray
      for (item <- batch) {
        for (i <- 0 until (item._2 * count).toInt) {
          temp(indexes(index)).append(item._1: _*)
          index += 1
        }
      }

      index = 0
    }

    temp
  //    temp.map(_.toArray)
  }

  def showTime(title: String, start: Long, end: Long): Unit = {
    printf("%s: %f\n", title, (end -start) * 1e-9)
  }

  def main(args: Array[String]): Unit = {
//    runSample(800, Array(7, 6, 5, 4, 3, 2), 300000)

//    testCount(800, Array(7, 6, 5, 4, 3, 2), 128*1024*1024)
    runSampleByStorage(800, Array(7, 6, 5, 4, 3, 2), 8*1024*1024)

  }

  def testCount(itemNum: Int, sizes: Array[Int], storageSize: Long): Unit = {
    val sbf = SupportBatchFactory(Array.range(0, itemNum))
    val batches = sbf.generateSupportBatches(sizes)
    val count = calculateCount(batches, storageSize)
    printf("count: %d", count)

  }

  def runSampleByStorage(itemNum: Int, sizes: Array[Int], storageSize: Long): Unit = {
    val sbf = SupportBatchFactory(Array.range(0, itemNum))
    val start = System.nanoTime()
    val batches = sbf.generateSupportBatches(sizes)
    val batchEnd = System.nanoTime()

    val count = calculateCount(batches, storageSize)

    val itemsets = generateFromBatch(batches, count)
    var genEnd = System.nanoTime()
    //    itemsets.slice(0, 10).foreach(r => printf("%s\n", r.toStr))
    showTime("BatchTime", start, batchEnd)
    showTime("Generate time", batchEnd, genEnd)
    val totalItem = itemsets.map(_.length.toLong).sum
    printf("total: %d\n", totalItem)
    printf("totalSize: %d, targetSize: %d\n", totalItem * 4, storageSize)
  }

  def runSample(itemNum: Int, sizes: Array[Int], count: Int): Unit = {
    val sbf = SupportBatchFactory(Array.range(0, itemNum))
    val start = System.nanoTime()
    val batches = sbf.generateSupportBatches(sizes)
    val batchEnd = System.nanoTime()
//    showTime("BatchTime", start, batchEnd)

//    val batch = batches(0)
//
//    for (item <- batch) {
//      printf("Batch: %s: %f\n", item._1.toStr, item._2)
//    }
//
//    printf("Isolated: %s: %f\n", batches(1)(0)._1.toStr, batches(1)(0)._2)

    val itemsets = generateFromBatch(batches, count)
    var genEnd = System.nanoTime()
//    itemsets.slice(0, 10).foreach(r => printf("%s\n", r.toStr))
    showTime("BatchTime", start, batchEnd)
    showTime("Generate time", batchEnd, genEnd)
    val totalItem = itemsets.map(_.length).sum
    printf("total: %d\n", totalItem)
  }
}
