
package org.apache.spark.datagen.associations

class BatchItem(_items: Array[Int], _support: Double) {
  var items = _items
  var support = _support

  def toTuple(): (Array[Int], Double) = (items, support)

}
