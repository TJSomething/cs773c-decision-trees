import weka.core.converters.CSVLoader
import weka.classifiers.trees.J48
import weka.filters.unsupervised.attribute.Discretize
import weka.core.Instance
import scala.math.sqrt
import scala.math.pow
import java.io.File
import weka.filters.Filter

object Main {
	def main(args: Array[String]) {
		val (filename, classCount) = try {
			(args(0), args(1).toInt)
		} catch {
			case _: NumberFormatException | _: ArrayIndexOutOfBoundsException => {
				println("Syntax:")
				println(if (args.length > 0) args(0) else "<program name>" +
				    " <CSV file> <number of bins to discretize class>")
				sys.exit()
			}
		}
		
		// Load data
		val loader = new CSVLoader
		loader.setSource(new File(filename))
		val data = loader.getDataSet()

		// Filter data
		val filter = new Discretize
		filter.setBins(classCount)
		filter.setUseEqualFrequency(true)
		filter.setAttributeIndices("9")
		filter.setInputFormat(data)

		val newData = Filter.useFilter(data, filter)

		// If we set a class for the data before filtering, the filtering won't
		// work
		data.setClassIndex(8)

		// We are classifying on column 9, age, which we just discretized
		newData.setClassIndex(8)

		// Learn from data
		val resultingTrees =
			for (i <- 0 to 10) yield {
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
			for (i <- 0 until 10) yield {
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
		val (bestTree, bestError) = resultingTrees zip rmsErrors minBy(_._2)
		println("Best tree:")
		println(bestTree)
		println("RMS Error: " + bestError toString)
	}
}
