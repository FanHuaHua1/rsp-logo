
package org.apache.spark.rsp

import org.apache.spark.Partition
import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD}

class SonPartitionCoalescer(index: Array[Int]) extends PartitionCoalescer with Serializable {

  override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
    require(maxPartitions == index.size)
    index.map(i => newGroup(parent.partitions(i)))
  }

  def newGroup(partition: Partition): PartitionGroup = {
    var group = new PartitionGroup()
    group.partitions += partition
    group
  }
}