
package org.apache.spark

import breeze.linalg.{DenseVector}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.algo.classification.{DecisionTrees, LinearRegression, RandomForest}
import org.apache.spark.datagen.clusters.{Cluster, ClusterGenerator, DistributionConstructors, GridShifter, Hyperplane, Shifters, SingleCluster, Types}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.rsp.{RspRDD, SonRDD}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.Row
import org.apache.spark.sql.RspContext._
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.linalg.{DenseVector => SparkDenseVector}
import org.apache.spark.ml.util.{MLWritable, MLWriter}
import org.apache.spark.rdd.{PartitionCoalescer, PartitionGroup, RDD}

import scala.collection.mutable.WrappedArray
import scala.util.Random
import spire.ClassTag
import smile.regression.{LinearModel, OLS}
import smile.data.{DataFrame => SmileDataFrame}
import smile.classification.{Classifier, SVM}
import smile.data.formula.Formula

object RSPCluster {

  val KB = 1024
  val MB = 1024*KB
  val GB = 1024*MB
  val TB = 1024*GB.toLong

  var AppName: String = null
  var MaxExecutors: Int = 20
//  var Conf: SparkConf = new SparkConf()

  lazy val conf: SparkConf = getConf()
  lazy val spark: SparkSession = getSpark()

    def main(args: Array[String]): Unit = {

      printf("Main args: %s\n", args.reduce((a, b) => a + " " + b))
      run(args)
    }

  def run(args: Array[String]): Unit = {
    if (args.size > 0) {
      args(0) match {
        case "gen" => onGenerate(args)
        case "rsp" => onRsp(args)
        case "subrsp" => onSubRsp(args)
        case "fit" => onFit(args)
        case "conf" => onConf(args)
        case "fitrsp" => onFitRSP(args)
        case "smile" => onFitSmile(args)
        case "plane" => onPlane(args)
        case "reg" => onReg(args)
        case "algo" => onAlgo(args)
        case "fitAlgo" => onFitAlgo(args)
        case "--executors" => {
//          MaxExecutors = args(1).toInt
          conf.set("spark.dynamicAllocation.maxExecutors", args(1))
          run(args.slice(2, args.size))
        }
        case "--conf" => {
          var confs = args(1).split(",")
          confs.foreach(c => {
            var kv = c.split("=")
            conf.set(kv(0), kv(1))
          })

          run(args.slice(2, args.size))
        }
        case _ => test(args)

      }

    }
  }

//  def setConf(args)
  def getConf(): SparkConf = {
    var sparkConf = new SparkConf()
    sparkConf.setMaster("yarn")
    sparkConf.set("spark.dynamicAllocation.enabled", "true")
    sparkConf.set("spark.shuffle.service.enabled", "true")
    sparkConf.set("spark.dynamicAllocation.minExecutors", "1")
    sparkConf.set("spark.executor.memory", "4g")
    sparkConf
  }

  def getSpark(): SparkSession = {
//    var builder = SparkSession.builder().master(
//      "yarn"
//    ).config(
//      "spark.dynamicAllocation.enabled", true
//    ).config(
//      "spark.dynamicAllocation.maxExecutors", MaxExecutors
//    ).config(
//      "spark.shuffle.service.enabled", true
//    ).config(
//      "spark.dynamicAllocation.minExecutors", 1
//    ).config(
//      "spark.executor.memory", "4g"
//    )

    var builder = SparkSession.builder().config(conf)

    if (AppName != null) {
      builder.appName(AppName)
    }

    builder.getOrCreate()
  }

  def onAlgo(args: Array[String]): Unit = {
    import org.apache.spark.algo.classification.{DecisionTrees, LinearRegression, RandomForest}
    var name = args(1)
    name match {
      case "DT" => DecisionTrees.runSpark(spark)
      case "LR" => LinearRegression.runSpark(spark)
      case "RF" => RandomForest.runSpark(spark)
    }
  }

  def onConf(args: Array[String]): Unit = {
    printf("AppName: %s\n", spark.sparkContext.appName)
    printf("Args: %s\n", args.reduce((a, b) => a + " " + b))
  }

  def test(args: Array[String]): Unit = {
    var oFile = decodeOriginFile("classification_20_2_0.42_4000_1T.parquet")
    printf(
      "File: %s\nD: %d\nC: %d\nS: %f\nP: %d\ns: %d\n",
      oFile.filename,
      oFile.dimension,
      oFile.clusters,
      oFile.scale,
      oFile.partitions,
      oFile.size
    )
  }

  case class OriginFile(filename: String,
                        dimension: Int = 0,
                        clusters: Int = 0,
                        scale: Double = 0,
                        partitions: Int = 0,
                        size: Long = 0)

  def decodeOriginFile(filename: String): OriginFile = {
    // filename: head_dimension_clusters_scale_partition_size.parquet
    var arguments = filename.split('_')
    var sizeStr = arguments(5).split('.')(0)
    var size = sizeStr(sizeStr.size-1) match {
      case 'G' => sizeStr.slice(0, sizeStr.size-1).toLong * GB
      case 'M' => sizeStr.slice(0, sizeStr.size-1).toLong * MB
      case 'K' => sizeStr.slice(0, sizeStr.size-1).toLong * KB
      case 'g' => sizeStr.slice(0, sizeStr.size-1).toLong * GB
      case 'k' => sizeStr.slice(0, sizeStr.size-1).toLong * KB
      case 'm' => sizeStr.slice(0, sizeStr.size-1).toLong * MB
      case 'T' => sizeStr.slice(0, sizeStr.size-1).toLong * TB
      case 't' => sizeStr.slice(0, sizeStr.size-1).toLong * TB
      case _ => sizeStr.toLong
    }

    OriginFile(
      filename,
      arguments(1).toInt,
      arguments(2).toInt,
      arguments(3).toDouble,
      arguments(4).toInt,
      size
    )
  }

    def normalConstructor(dimension: Int, scale: Double): Array[Types.DistributionConstructor] = {
      Array.range(0, dimension).map(
        _ => DistributionConstructors.gaussian(0, scale)
      )
    }

    def decodeOutputFilename(filename: String): (Int, Long) = {
      // filename: name_partition_size.parquet
      var arguments = filename.split(Array('_', '.'))
      var sizeStr = arguments(2)
      var size = sizeStr(sizeStr.size-1) match {
        case 'G' => sizeStr.slice(0, sizeStr.size-1).toLong * GB
        case 'M' => sizeStr.slice(0, sizeStr.size-1).toLong * MB
        case 'K' => sizeStr.slice(0, sizeStr.size-1).toLong * KB
        case 'g' => sizeStr.slice(0, sizeStr.size-1).toLong * GB
        case 'k' => sizeStr.slice(0, sizeStr.size-1).toLong * KB
        case 'm' => sizeStr.slice(0, sizeStr.size-1).toLong * MB
        case 'T' => sizeStr.slice(0, sizeStr.size-1).toLong * TB
        case 't' => sizeStr.slice(0, sizeStr.size-1).toLong * TB
        case _ => sizeStr.toLong
      }

      return (arguments(1).toInt, size)
    }

    def onGenerate(args: Array[String]): Unit = {
      var outputFile = args(1)
      printf("Output: %s\n", outputFile)

      var oFile = decodeOriginFile(outputFile)

      printf("Partitions: %d\n", oFile.partitions)
      printf("Size: %d\n", oFile.size)
      printf("Dimension: %d\n", oFile.dimension)
      printf("clusters: %d\n", oFile.clusters)
      printf("scale: %f\n", oFile.scale)
      var df = generateSample(
        spark,
        oFile.dimension,
        oFile.clusters,
        oFile.scale,
        oFile.partitions,
        oFile.size,
        outputFile)
      df.show()
    }

    def generateSample(spark: SparkSession,
                       dimension: Int,
                       clusters: Int,
                       scale: Double,
                       partitions: Int,
                       sizeInBytes: Long,
                       outputFile: String
                      ): DataFrame = {

      var countsPerCluster = sizeInBytes / (8*dimension + 4) / clusters
      var counts = (countsPerCluster / partitions).asInstanceOf[Int]
      var shifter = getShifter(dimension)
      var scs = Array.range(0, clusters).map(
        a => SingleCluster(
          normalConstructor(dimension, scale)
        ).asInstanceOf[Cluster]
      )

      var cg = ClusterGenerator(
        dimension = dimension,
        clusters = scs,
        shifter = shifter
      )
      printf("Count/partition/cluster: %d\n", counts)
      var df = cg.generate(spark, partitions, Array.range(0, clusters).map(_ => counts))
      printf("Generate to parquet: %s\n", outputFile)
      spark.time(df.write.parquet(outputFile))
      return df

    }

  def onPlane(args: Array[String]): Unit = {
    var outputFile = args(1)
    printf("Output: %s\n", outputFile)

    var oFile = decodeOriginFile(outputFile)

    printf("Partitions: %d\n", oFile.partitions)
    printf("Size: %d\n", oFile.size)
    printf("Dimension: %d\n", oFile.dimension)
    printf("clusters: %d\n", oFile.clusters)
    printf("noise: %f\n", oFile.scale)

    generatePlane(oFile.dimension, oFile.scale, oFile.partitions, oFile.size, outputFile)
  }

  def generatePlane(dimension: Int,
                    noise: Double,
                    partitions: Int,
                    sizeInBytes: Long,
                    outputFile: String): Unit = {
    var vectorDF = spark.read.parquet("HyperplaneVectors.parquet")
    var vectors = vectorDF.rdd.map(
      r => (r.getInt(0), r.get(1).asInstanceOf[SparkDenseVector].asBreeze.toDenseVector)
    ).collect().toMap

    var vector = vectors(dimension)
    var distance = vector(vector.size - 1)
    var countsPerCluster = sizeInBytes / (8*dimension + 4)
    var counts = (countsPerCluster / partitions).asInstanceOf[Int]

    var clusters = Array(
      Hyperplane.getHyperplaneCluster(
        Array.range(0, dimension).map(
          _ => DistributionConstructors.uniform(-1, 1)
        ),
        vector.map(_ / distance),
        distance,
        DistributionConstructors.gaussian(0, noise)
      )
    )

    var cg = ClusterGenerator(
      dimension,
      clusters,
      Shifters.zeros()
    )

    printf("Count/partition/cluster: %d\n", counts)
    var df = cg.generate(spark, partitions, Array(counts))
    printf("Generate to parquet: %s\n", outputFile)
    df.write.parquet(outputFile)

  }

  def onReg(args: Array[String]): Unit = {
    if (args(1) == "rsp") {
      regressionRSP(args(2))
    } else {

      regression(args(1))
    }
  }

  def regression(filename: String): Unit = {
    import spark.implicits._
    val df = spark.read.parquet(filename)
    val trainDF = df.rdd.map(row => {
      var vector = row.get(1).asInstanceOf[SparkDenseVector].values;
      (vector(vector.size-1), new SparkDenseVector(vector.slice(0, vector.size-1)))
    }).toDF("label", "features")
    import org.apache.spark.ml.regression.LinearRegression
    val lr = new LinearRegression().setMaxIter(20).setRegParam(0.05).setElasticNetParam(0.05)
    val lrModel = lr.fit(trainDF)
    printf(
      "coefficients: %s\n",
      lrModel.coefficients.toString
    )
    printf("intercept: %f\n", lrModel.intercept)
//    var summary = lrModel.summary
//    sum
  }

  def regressionRSP(filename: String): Unit = {
    import spark.implicits._
    val df = spark.rspRead.parquet(filename).getSubDataset(1)
    var training = df.rdd.map(row => row.get(1).asInstanceOf[SparkDenseVector].toArray)
    var models = training.mapPartitions(smileRegression).collect()
//    var models = smileRegression(training.sample(false, 0.1).collect().iterator)
    for (model <- models) {
      printf("%s\n", model.toString)
    }
  }

  def smileRegression(it: Iterator[Array[Double]]): Iterator[LinearModel] = {
    val sample = it.toArray
    val names = Array.range(1, sample(0).size + 1).map("x" + _.toString)
    val df = SmileDataFrame.of(sample, names: _*)
    val columns = df.names()
    val formula = Formula.of(columns(columns.size-1), columns.slice(0, columns.size-1): _*)
    val linearModel = OLS.fit(formula, df)
    Iterator(linearModel)
  }

    def getShifter(dimension: Int): Types.ClusterShiftFunc = {

//      GridShifter.grid()
      dimension match {
        case 20 => (numbers: Int, dimension: Int) => Array(
          DenseVector(
            0.07231505, 0.14550194, 0.17010658, 0.41748683, 0.45338776,
            0.24121279, 0.01608654, 0.36348614, 0.25394233, 0.23053118,
            0.44549768, 0.31422324, 0.23071721, 0.42843829, 0.28006534,
            0.07198154, 0.48340346, 0.05189738, 0.2410514, 0.24165247
          ),
          DenseVector(
            0.11808569, 0.21891885, 0.47693982, 0.43086218, 0.30976608,
            0.08831673, 0.49430253, 0.11723838, 0.24033244, 0.18433353,
            0.04137646, 0.02806667, 0.29535499, 0.05468401, 0.19828239,
            0.13736125, 0.47304012, 0.34852589, 0.44426084, 0.83049285
          )
        )
        case 50 => (numbers: Int, dimension: Int) => Array(
          DenseVector(
            0.250686, 0.142326, 0.190033, 0.072311, 0.457978,
            0.358443, 0.476069, 0.005067, 0.097645, 0.082783,
            0.280516, 0.378278, 0.122328, 0.409079, 0.103781,
            0.040576, 0.242090, 0.341708, 0.408516, 0.380999,
            0.307336, 0.091583, 0.143618, 0.227071, 0.017078,
            0.479626, 0.353369, 0.429018, 0.419288, 0.158281,
            0.019275, 0.470716, 0.075437, 0.104568, 0.438518,
            0.096431, 0.167925, 0.043191, 0.133598, 0.344227,
            0.409758, 0.137654, 0.432037, 0.003987, 0.476458,
            0.004821, 0.246673, 0.077828, 0.186476, 0.117277
          ),
          DenseVector(
            0.374666, 0.283729, 0.180982, 0.034713, 0.449087,
            0.335948, 0.282857, 0.302961, 0.241231, 0.350389,
            0.489892, 0.311905, 0.158041, 0.206635, 0.029994,
            0.111552, 0.443252, 0.274238, 0.163158, 0.204453,
            0.183921, 0.152557, 0.370403, 0.429132, 0.062448,
            0.459963, 0.072475, 0.027525, 0.334263, 0.491908,
            0.059192, 0.372031, 0.160413, 0.325144, 0.195736,
            0.095076, 0.308292, 0.455574, 0.039429, 0.196983,
            0.273505, 0.483596, 0.121775, 0.315639, 0.033889,
            0.229093, 0.259526, 0.079366, 0.059543, 0.571722
          )
        )
        case 100 => (numbers: Int, dimension: Int) => Array(
          DenseVector(
            0.238499, 0.037805, 0.048724, 0.123271, 0.296273,
            0.118141, 0.273874, 0.440173, 0.424682, 0.309454,
            0.101449, 0.195164, 0.360524, 0.327251, 0.342548,
            0.448836, 0.400749, 0.463592, 0.092495, 0.182218,
            0.365328, 0.351054, 0.495940, 0.162062, 0.216753,
            0.325691, 0.117642, 0.313780, 0.413422, 0.310068,
            0.486193, 0.144609, 0.024589, 0.219067, 0.164347,
            0.411417, 0.325426, 0.126837, 0.250813, 0.192542,
            0.396083, 0.426030, 0.288257, 0.270208, 0.129214,
            0.199427, 0.413976, 0.184282, 0.339453, 0.404557,
            0.458750, 0.389855, 0.079252, 0.041311, 0.459836,
            0.093411, 0.442382, 0.255599, 0.196500, 0.467820,
            0.349209, 0.353468, 0.362499, 0.191734, 0.461241,
            0.155133, 0.070440, 0.406092, 0.479983, 0.162400,
            0.018135, 0.212037, 0.308271, 0.096859, 0.363534,
            0.204274, 0.368928, 0.275337, 0.052529, 0.422413,
            0.099211, 0.490934, 0.004720, 0.221284, 0.498984,
            0.165154, 0.402953, 0.143540, 0.170529, 0.091404,
            0.409570, 0.149775, 0.076635, 0.248944, 0.452929,
            0.372749, 0.288709, 0.189147, 0.028026, 0.355263
          ),
          DenseVector(
            0.332673, 0.209951, 0.284120, 0.469457, 0.343568,
            0.202784, 0.392555, 0.457812, 0.220427, 0.257005,
            0.259911, 0.319810, 0.124907, 0.265970, 0.185595,
            0.004079, 0.302369, 0.164762, 0.384868, 0.327388,
            0.445670, 0.322337, 0.326525, 0.263109, 0.164381,
            0.119262, 0.194794, 0.463744, 0.327149, 0.027786,
            0.166399, 0.382263, 0.029800, 0.092816, 0.006220,
            0.190701, 0.050945, 0.410515, 0.377804, 0.238468,
            0.059621, 0.288621, 0.475682, 0.327604, 0.176765,
            0.265175, 0.018470, 0.312880, 0.095707, 0.383144,
            0.107471, 0.217781, 0.264290, 0.492530, 0.000631,
            0.200498, 0.282565, 0.447731, 0.011213, 0.211246,
            0.027906, 0.017115, 0.474786, 0.439406, 0.119087,
            0.164477, 0.227353, 0.402878, 0.395708, 0.135796,
            0.238308, 0.445870, 0.201157, 0.184248, 0.438169,
            0.485268, 0.453276, 0.139773, 0.422942, 0.318319,
            0.489974, 0.023606, 0.054290, 0.347758, 0.311924,
            0.310346, 0.168890, 0.335825, 0.353175, 0.122245,
            0.286484, 0.366814, 0.277893, 0.356997, 0.297109,
            0.300051, 0.051046, 0.450175, 0.363208, 0.919168
          )
        )
        case _ => GridShifter.grid()
      }
    }

  def onFitSmile(args: Array[String]): Unit = {
    fitSmile(args(1))
  }

  def fitSmile(filename: String, fraction: Double = 0.9): Unit = {
    printf("Fit smile svm: %s\n", filename)
    var df = spark.rspRead.parquet(filename)

    var sRdd = df.rdd.map(
      r => (transferLabel(r.getInt(0)), r.get(1).asInstanceOf[SparkDenseVector].toArray)
    )

    var rdds = sRdd.randomSplit(Array(fraction, 1 - fraction))

    var clsRdd = rdds(0).mapPartitions(mapPartitionSmileSvm)
    var classifer = clsRdd.first()
    var prediction = rdds(1).map(item => (item._1, classifer.predict(item._2)))

    var accuracy = prediction.filter(
      r => r._1 == r._2
    ).count().toDouble / prediction.count().toDouble
    printf("%s Accuracy: %f\n", filename, accuracy)


  }

  def transferLabel(l: Int): Int = {
    if (l == 1) {
      return 1
    } else {
      return -1
    }
  }
//  def smileSvmPredict(x: Array[Double], classifer: Classifier[Array[Double]]): Unit = {
//    classifer.predict(x)
//  }

  def mapPartitionSmileSvm(it: Iterator[(Int, Array[Double])]):
  Iterator[Classifier[Array[Double]]] = {
    var arr = it.toArray
    var x = arr.map(_._2)
    var y = arr.map(_._1)
    Iterator(SVM.fit(x, y, 5, 1E-3))

  }

  def onRsp(args: Array[String]): Unit = {
    var dfFile = args(1)
    var outputFile = args(2)
    var arguments = outputFile.split(Array('_', '.'))
    var rspPartitions = arguments(1).toInt
    var df = spark.rspRead.parquet(dfFile)
    rsp(spark, df, rspPartitions, outputFile)
  }

  def rsp(spark: SparkSession,
          df: DataFrame,
          partitions: Int,
          outputFile: String): Unit = {
    printf("df count: %d\n", df.count())
    printf("rsp partition: %d\n", partitions)
    printf("outputFile: %s\n", outputFile)
    printf("start rsp convert\n")
    var rdd = df.rdd.toRSP(partitions)

    var rspDF = spark.createDataFrame(rdd, df.schema)
    spark.time(rspDF.write.parquet(outputFile))
    rspDF.show()
    printf("rspDF count: %d\n", rspDF.count())

  }

  def onSubRsp(args: Array[String]): Unit = {
    var dfFile = args(1)
    var outputFile = args(2)
    var arguments = outputFile.split(Array('_', '.'))
    var rspPartitions = arguments(1).toInt
    var df = spark.rspRead.parquet(dfFile)
    subRSP(spark, df, rspPartitions, outputFile)
  }

  implicit  class ArrayFunc[T](arr: Array[T]) extends  Serializable {

    def toStr(): String = {
      arr.map(_.toString).reduce(_ + " " + _)
    }
  }

  def subRSP(spark: SparkSession,
             df: DataFrame,
             partitions: Int,
             outputFile: String): Unit = {
    printf("df count: %d\n", df.count())
    printf("rsp sub partition: %d\n", partitions)
    printf("outputFile: %s\n", outputFile)
    printf("start rsp convert\n")
    var index = Random.shuffle(List.range(0, df.rdd.getNumPartitions)).take(partitions).toArray
//    var subRdd = getSubRdd(df.rdd, index)
    var subRdd = df.rdd.getSubPartitions(index)
    var rdd = subRdd.toRSP(partitions)
    var rspDF = spark.createDataFrame(rdd, df.schema)
    spark.time(rspDF.write.parquet(outputFile))
    rspDF.show()
    printf("rspDF count: %d\n", rspDF.count())

  }

  class SonPartitionCoalescer(index: Array[Int]) extends PartitionCoalescer with Serializable {

    override def coalesce(maxPartitions: Int, parent: RDD[_]): Array[PartitionGroup] = {
      require(maxPartitions == index.size)
      index.map(i => newGroup(parent.partitions(i)))
//      parent.partitions.map(newGroup(_))
    }

    def newGroup(partition: Partition): PartitionGroup = {
      var group = new PartitionGroup()
      group.partitions += partition
      group
    }
  }

  class SonPartition(_index: Int, prev: Partition) extends Partition {
    /**
     * Get the partition's index within its parent RDD
     */
    override def index: Int = _index

    override def hashCode(): Int = prev.hashCode()

    override def equals(other: Any): Boolean = prev.equals(other)

  }

  implicit  class RDDFunc[T: ClassTag](rdd: RDD[T]) extends  Serializable {

    def getSubPartitions(index: Array[Int]): RspRDD[T] = {
      new RspRDD[T](
        rdd.coalesce(index.size, false, Option(new SonPartitionCoalescer(index)))
      )
    }
  }

  implicit  class RspRDDFunc[T: ClassTag](rdd: RspRDD[T]) extends  Serializable {

    def getSubPartitions(index: Array[Int]): RspRDD[T] = {
      new RspRDD[T](
        rdd.coalesce(index.size, false, Option(new SonPartitionCoalescer(index)))
      )
    }
  }

  def getSubRdd[T: ClassTag](rdd: RDD[T], index: Array[Int]): RDD[T] = {
    rdd.coalesce(index.size, false, Option(new SonPartitionCoalescer(index)))
  }

  def onFit(args: Array[String]): Unit = {
    var fraction: Double = 1

    if (args.size>2) {
      fraction = args(2).toDouble
    }
    fit(spark, args(1), fraction)
  }

  def fit(spark: SparkSession, filename: String, fraction: Double): Unit = {

    var df = spark.read.parquet(filename)
    if (fraction < 1) {
      df = df.sample(fraction)
    }

    var dfs = df.randomSplit(Array(0.9, 0.1))
    var lsvc = new LinearSVC().setMaxIter(20).setRegParam(0.05)
    printf("start train\n")
    var model = spark.time(lsvc.fit(dfs(0)))
    printf("train finished\n")
    var predictions = model.transform(dfs(1))
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

    var accuracy = evaluator.evaluate(predictions)
    printf("%s Accuracy: %f\n", filename, accuracy)

  }

  /**
   *
   * @param args: fitrsp [sourceFile] [partFile] [fraction] [sizes]
   */
  def onFitRSP(args: Array[String]): Unit = {
    var sourceFile = args(1)
    var partitionFile = args(2)
    var fraction = args(3).toDouble
    var sizes = args.slice(4, args.size).map(_.toInt)

    fitRSP(spark, sourceFile, partitionFile, fraction, sizes)

  }

  def fitRSP(spark: SparkSession,
             sourceFile: String,
             partitionFile: String,
             fraction: Double,
             sizes: Array[Int]): Unit = {

    printf("fitRSP: sourceFile = %s\n", sourceFile)
    printf("fitRSP: partitionFile = %s\n", partitionFile)
    printf("fitRSP: fraction = %f\n", fraction)
    var rdf = spark.rspRead.parquet(sourceFile)
    var pdf = spark.read.parquet(partitionFile)
    var parts = pdf.collect().map(
      r => (r.getInt(0), r.get(1).asInstanceOf[WrappedArray[Int]].toArray)
    )
    var jobs: Array[(Int, Array[Int])] = null
    if (sizes.size > 0) {
      printf("fitRSP: sizes = %s\n", sizes.map(_.toString).reduce(_ + ", " + _))
      var sizeMapper = parts.toMap
      jobs = sizes.filter(s => sizeMapper.contains(s)).map(s => (s, sizeMapper(s)))
    } else {
      jobs = parts
    }

    for ((size, partitions) <- jobs) {
      var trainName = "train(size=%d)".format(size)
//      var partitions = sizeMapper(size)
//      var rdd = getSubRdd(rdf.rdd, partitions)
      var rdd = rdf.rdd.getSubPartitions(partitions)

      var df = spark.createDataFrame(rdd, rdf.schema)

      var dfs: Array[DataFrame] = null
      if (fraction >= 1 ) {
        dfs = df.randomSplit(Array(0.9, 0.1))
      } else {
        dfs = df.randomSplit(Array(fraction, 1 - fraction))
      }

//      var dfs = df.randomSplit(Array(fraction, 1 - fraction))
      var lsvc = new LinearSVC().setMaxIter(20).setRegParam(0.05)
      printf("%s start\n", trainName)
      var model = spark.time(lsvc.fit(dfs(0)))
      printf("%s finished\n", trainName)
      var predictions = model.transform(dfs(1))
      val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

      var accuracy = evaluator.evaluate(predictions)
      printf("%s Accuracy: %f\n", trainName, accuracy)


    }

  }

  def fitAlgo(spark: SparkSession,
              algo: String,
              sourceFile: String,
              partitionFile: String,
              fraction: Double,
              sizes: Array[Int]): Unit = {

    printf("fitAlgo: algorithm = %s\n", algo)
    printf("fitAlgo: sourceFile = %s\n", sourceFile)
    printf("fitAlgo: partitionFile = %s\n", partitionFile)
    printf("fitAlgo: fraction = %f\n", fraction)
    var rdf = spark.rspRead.parquet(sourceFile)
    var pdf = spark.read.parquet(partitionFile)
    var parts = pdf.collect().map(
      r => (r.getInt(0), r.get(1).asInstanceOf[WrappedArray[Int]].toArray)
    )
    var jobs: Array[(Int, Array[Int])] = null
    if (sizes.size > 0) {
      printf("fit algo: sizes = %s\n", sizes.map(_.toString).reduce(_ + ", " + _))
      var sizeMapper = parts.toMap
      jobs = sizes.filter(s => sizeMapper.contains(s)).map(s => (s, sizeMapper(s)))
    } else {
      jobs = parts
    }

    for ((size, partitions) <- jobs) {
      var trainName = "train(size=%d)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
      var modelName = "%s_%s_%d_%s.ml".format(algo, sourceFile, size, beginTime)
      //      var partitions = sizeMapper(size)
      //      var rdd = getSubRdd(rdf.rdd, partitions)
      var rdd = rdf.rdd.getSubPartitions(partitions)

      var df = spark.createDataFrame(rdd, rdf.schema)

      algo match {
        case "DT" => runAlgo(trainName, modelName, df, fraction,
          DecisionTrees.SparkDecisionTreesClasscification)
        case "LR" => runAlgo(trainName, modelName, df, fraction,
          LinearRegression.SparkLinearRegressionClassification)
        case "RF" => runAlgo(trainName, modelName, df, fraction,
          RandomForest.SparkRandomForestClasscification)
      }
    }

  }

  def onFitAlgo(args: Array[String]): Unit = {
    var algo = args(1)
    var sourceFile = args(2)
    var partitionFile = args(3)
    var fraction = args(4).toDouble
    var sizes = args.slice(5, args.size).map(_.toInt)

    fitAlgo(spark, algo, sourceFile, partitionFile, fraction, sizes)

  }

  def runAlgo(trainName: String, modelName: String,
              df: DataFrame, fraction: Double,
              function: (DataFrame, Double) => (MLWritable, Double, Double)): Unit = {
    printf("%s start\n", trainName)
    var (model, duration, accuracy) = function(df, fraction)
    printf("%s finished\n", trainName)
    printf("Time spend: %f\n", duration)
    printf("%s Accuracy: %f\n", trainName, accuracy)
    model.write.save(modelName)
  }

  def onFitLogo(args: Array[String]): Unit = {
    var algo = args(1)
    var sourceFile = args(2)
    var partitionFile = args(3)
    var subs = args(4).toDouble
    var tests = args(5).toInt
    var predicts = args(6).toInt
    var sizes = args.slice(5, args.size).map(_.toInt)
  }

  def fitLogo(spark: SparkSession,
              algo: String,
              sourceFile: String,
              partitionFile: String,
              subs: Double, tests: Int, predicts: Int,
              sizes: Array[Int]): Unit = {
    var rdf = spark.rspRead.parquet(sourceFile)
    var pdf = spark.read.parquet(partitionFile)

    var parts = pdf.collect().map(
      r => (r.getInt(0), r.get(1).asInstanceOf[WrappedArray[Int]].toArray)
    )
    var jobs: Array[(Int, Array[Int])] = null
    if (sizes.size > 0) {
      printf("fit algo: sizes = %s\n", sizes.map(_.toString).reduce(_ + ", " + _))
      var sizeMapper = parts.toMap
      jobs = sizes.filter(s => sizeMapper.contains(s)).map(s => (s, sizeMapper(s)))
    } else {
      jobs = parts
    }

    for ((size, partitions) <- jobs) {
      var trainName = "train(size=%d)".format(size)
      var beginTime = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYYYMMdd-HHmmss"))
//      var modelName = "%s_%s_%d_%s.ml".format(algo, sourceFile, size, beginTime)
      //      var partitions = sizeMapper(size)
      //      var rdd = getSubRdd(rdf.rdd, partitions)
      var trainParts = (partitions.length * subs).toInt
      var testParts = trainParts + tests
      var predictParts = testParts + predicts
      var trainRdd = rdf.rdd.getSubPartitions(partitions.slice(0, trainParts))
      var testRdd = rdf.rdd.getSubPartitions(partitions.slice(trainParts, testParts))
      var predictRdd = rdf.rdd.getSubPartitions(partitions.slice(testParts, predictParts))

    }
  }

  def predictModel[U: ClassTag](m: U, data: Array[Row]): Double = {
    Random.nextDouble()
  }

//  def aggResults(tag: Long, groups: Iterable[(Long, Long, Double)]): (Long, Double) = {
  def aggResults(item: (Long, Iterable[(Long, Long, Double)])): (Long, Double) = {
    var acc = item._2.map(_._3).sum / groups.size
    return (item._1, acc)
  }

  def runLogo[U: ClassTag](trainRdd: RDD[Row], testRdd: RDD[Row], predictRdd: RDD[Row],
              trainer: (Iterator[Row]) => Iterator[U]): Unit = {
    var modelRdd = trainRdd.mapPartitions(trainer).zipWithIndex()

    var scoreRdd = modelRdd.cartesian(
      testRdd.glom().zipWithIndex()
    ).map(item => (item._1._2, item._2._2, predictModel(item._1._1, item._2._1)))

    var groupRdd = scoreRdd.groupBy(_._1)
    var aggRDD = groupRdd.map(aggResults)
    
  }


}
