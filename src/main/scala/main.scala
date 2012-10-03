import weka.core.converters.CSVLoader
import weka.classifiers.trees.J48
import weka.filters.unsupervised.attribute.Discretize
import weka.core.Instance
import scala.math.sqrt
import scala.math.pow
import java.io.File
import weka.filters.Filter
import java.util.Random
import weka.core.Instances

object Main {
	val help = "Syntax: " + 
	           "<program> <CSV file> <# of bins>\n\n" + 
	           "<CSV file> must be formatted with the names of the " +
	           "sample attributes\nas the first row. The last column is " +
	           "expected to be a numeric attribute\nthat will be " +
	           "divided into <# of bins> approximately equally-sized bins.\n" +
	           "These bins will be used as classes for the samples from " +
	           "<CSV file>."
	
	//val trials = 300
	
	def main(args: Array[String]) = {
		val (filename, classCount) = try {
			(args(0), args(1).toInt)
		} catch {
			case _: NumberFormatException|_: ArrayIndexOutOfBoundsException => {
				println(help)
				println
				sys.exit()
			}
		}
		
		// Load data
		val loader = new CSVLoader
		loader.setSource(new File(filename))
		val data = loader.getDataSet()
		
		/* Stuff for testing for the best number of bins
		 * val errors = for (i <- 0 until trials) yield {
			data.randomize(new Random)
			buildTree(data, classCount)._2
		}
		val mean = errors.sum / trials
		val stdDev =
		  sqrt((for (error <- errors) yield pow(mean - error,2)).sum
		      / (trials-1)) 
		println(mean.toString + "," + stdDev.toString )*/
		
		val (tree, error) =
		  (for (i <- 0 until 10) yield {
			data.randomize(new Random)
			buildTree(data, classCount)
		   }).minBy(_._2)
		
		println(tree)
		println("RMS error: " + error.toString)
	}
	
	def buildTree(data: Instances, classCount: Int) = {
		val classCol = data.numAttributes
		
		// Discretize data
		data.setClassIndex(-1) // If there is a class, then this fails
		val filter = new Discretize
		filter.setBins(classCount)
		filter.setUseEqualFrequency(true)
		filter.setAttributeIndices( classCol toString )
		filter.setInputFormat(data)

		val newData = Filter.useFilter(data, filter)

		// If we set a class for the data before filtering, the filtering won't
		// work
		data.setClassIndex(classCol-1)

		// We are classifying on column 9, age, which we just discretized
		newData.setClassIndex(classCol-1)

		// Learn from data
		val resultingTrees =
			for (i <- 0 until 1) yield {
				val tree = new J48
				tree.setReducedErrorPruning(true)
				tree.buildClassifier(newData.trainCV(10, i))
				tree
			}

		// Given a class string representing a discretized range, yield the
		// approximate center, unless one side of the class is infinity. In that case,
		// the other bound is considered the center of the class.
		def classCenter(classString: String) = {
			val rangeRegex =
				"""'\((-inf|-?\d+(?:\.\d*)?)-(-?\d+(?:\.\d*)?|inf)(?:[\)\]])'""".r
			val rangeRegex(loString, hiString) = classString
			if (loString == "-inf")
				if (hiString == "inf")
					0.0
				else
					hiString toDouble
			else
				if (hiString == "inf")
					loString toDouble
				else
					(loString.toDouble + hiString.toDouble)/2.0
		}

		// Evaluate results
		val rmsErrors =
			for (i <- 0 until 1) yield {
				val testSet = newData.testCV(10, i)
				val rawTestSet = data.testCV(10, i)
				// Calculate the root-mean-square of test set #i
				sqrt((for (j <- 0 until testSet.numInstances) yield {
						// Predict a value
						val sample = testSet.instance(j)
						val rings = sample.classAttribute

						val predictedValue =
								classCenter(
									rings.value(
										resultingTrees(i).classifyInstance(
										    sample).toInt))

						// Actual value
						val actualValue = data.instance(j).value(
						    data.classAttribute)
						
						// Square error
						pow(actualValue - predictedValue, 2)
					}).sum / testSet.numInstances)
			}

		// Find the most accurate tree and tell the world
		resultingTrees zip rmsErrors minBy(_._2)
	}
}
