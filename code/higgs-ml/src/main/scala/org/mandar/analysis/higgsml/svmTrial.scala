package org.mandar.analysis.higgsml

/**
 * Created by mandar on 13/6/14.
 */

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.regression.{GeneralizedLinearModel, LabeledPoint}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.{OrderedRDDFunctions, RDD}
import org.apache.spark.mllib.regression.LabeledPoint


object svmTrial {
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName("SVM Trial")
    val sc = new SparkContext(conf)
    val data = loadfromFile(sc, "hdfs:///higgs-ml/training-weka.csv")
    
    /*val matdata = new RowMatrix(data.map{vec => vec.features})
    val pc: Matrix = matdata.computePrincipalComponents(10)
    val projected: RowMatrix = matdata.multiply(pc)
    val finalproj = projected.rows.zip(data.map{vec => vec.label}).map{x => new LabeledPoint(x._2,x._1)}
    */

    // Split data into training (60%) and test (40%).
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the SVM model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    /*model.clearThreshold()
    val testdata = loadfromTestFile(sc, "hdfs:///higgs-ml/test-weka.csv")
    val result:RDD[(Double, (Int, String))] = testdata.map(
      event =>
      {
        val eventArr = event.toArray
        val features = Vectors.dense(eventArr.slice(1, eventArr.length-1))
        val score = model.predict(features)
        val label = if (score >= 0) "s" else "b"
        (score, (eventArr(0).toInt, label))

      }
    )*/

    def scoreAndLabels[T <: GeneralizedLinearModel](model : T) = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    /*val ord = new OrderedRDDFunctions[Double, (Int, String), (Double, (Int, String))](result)
    val sortedRes = ord.sortByKey(false).zipWithIndex()

    val finalRes = sortedRes.map(tuple => (tuple._1._2._1, tuple._2, tuple._1._2._2))

    finalRes.saveAsTextFile("hdfs:///higgs-ml/test-result")*/

    // Get evaluation metrics.
    val svmmetrics = new BinaryClassificationMetrics(scoreAndLabels(model))
    val auROC = svmmetrics.areaUnderROC()
    val auPR = svmmetrics.areaUnderPR()

    println("Area under ROC for SVM = " + auROC)
    println("Area under PR for SVM = " + auPR)

    val newmodel = LogisticRegressionWithSGD.train(training, numIterations)
    val logmetrics = new BinaryClassificationMetrics(scoreAndLabels(newmodel))
    val logauROC = logmetrics.areaUnderROC()
    val logauPR = logmetrics.areaUnderPR()

    println("Area under ROC for Logistic Regression = " + logauROC)
    println("Area under PR for Logistic Regression = " + logauPR)
    
  }

  def getLabel(arg: Char):Double = {
    if(arg == 's')
      1.0
    else
      0.0

  }

  def loadfromFile(sc: SparkContext, dir: String): RDD[LabeledPoint] = {
    sc.textFile(dir).filter{ line =>
      val parts = line.split(',')
      if (parts(0).toString == "EventId"){
        false
      } else {
        true
      }
    }.map { line =>
      val parts = line.split(',')
      val label = getLabel(parts(parts.length - 1).charAt(0))
      val truncated_parts = parts.slice(1, parts.length - 3)
      val features = Vectors.dense(truncated_parts.map{x:String => 
        if(x == "?") "0.0d".toDouble else x.toDouble })
      LabeledPoint(label, features)
    }
  }

  def loadfromTestFile(sc: SparkContext, dir: String): RDD[Vector] = {
    sc.textFile(dir).filter{ line =>
      val parts = line.split(',')
      if (parts(0).toString == "EventId"){
        false
      } else {
        true
      }
    }.map { line =>
      val parts = line.split(',')
      val truncated_parts = parts
      Vectors.dense(truncated_parts.map{x:String =>
        if(x == "?") "0.0d".toDouble else x.toDouble })
    }
  }

}
