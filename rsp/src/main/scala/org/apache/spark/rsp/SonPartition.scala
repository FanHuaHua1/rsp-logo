
package org.apache.spark.rsp

import org.apache.spark.Partition

class SonPartition(_index: Int, prev: Partition) extends Partition {
  /**
   * Get the partition's index within its parent RDD
   */
  override def index: Int = _index

  override def hashCode(): Int = prev.hashCode()

  override def equals(other: Any): Boolean = prev.equals(other)

}
