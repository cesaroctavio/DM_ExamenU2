//Librerias
import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.Column
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
////////////////////////////////////////////////////////////////////////////////////////
//Inicio de el archivo de Iris.csv
val spark = SparkSession.builder.master("local[*]").getOrCreate()

val df = spark.read.option("inferSchema","true").csv("Iris.csv").toDF(
  "SepalLength",
  "SepalWidth",
  "PetalLength",
  "PetalWidth",
  "class"
)

val newcol = when($"class".contains("Iris-setosa"), 1.0).
  otherwise(when($"class".contains("Iris-virginica"), 3.0).
  otherwise(2.0))

val newdf = df.withColumn("etiqueta", newcol)

newdf.select("etiqueta",
  "SepalLength",
  "SepalWidth",
  "PetalLength",
  "PetalWidth",
  "class").show(150, false)
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Limpieza de los datos
//Juntando el data
val assembler = new VectorAssembler()  .setInputCols(Array(
  "SepalLength",
  "SepalWidth",
  "PetalLength",
  "PetalWidth",
  "etiqueta")).setOutputCol("features")
//Transforming los data en features
val features = assembler.transform(newdf)
features.show(5)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Acomodar los labels, asi para añadir metadata para la columna label.
val labelIndexer = new StringIndexer().setInputCol("class").setOutputCol("indexedLabel").fit(features)
println(s"Found labels: ${labelIndexer.labels.mkString("[", ", ", "]")}")

//Se hace una categorizacion de las caracterisiticas de features y de ahi de indexan
val featureIndexer = new VectorIndexer().setInputCol("features")
.setOutputCol("indexedFeatures")
.setMaxCategories(4)
.fit(features)

//Se crean variables de entrenamiento , se hacen un test del entrenamineto
val splits = features.randomSplit(Array(0.6, 0.4))
val trainingData = splits(0)
val testData = splits(1)

// Se establece la estructura array sobre la red neuronal
val layers = Array[Int](5, 5, 5, 3)
/////////////////////////////////////////////////////////////////////////////////////////////////
// se crea una variable llamamdo trainer(entrenador) y se ingresan los parametros
val trainer = new MultilayerPerceptronClassifier()
.setLayers(layers).setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setBlockSize(128)
.setSeed(System.currentTimeMillis)
.setMaxIter(200)

//convierte los labels indexado a su forma normal de nuevo
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

//Los indexados de encadenan y la  MultilayerPerceptronClassifier en una  Pipeline.
val pipeline = new Pipeline().setStages(Array(
  labelIndexer,
  featureIndexer,
  trainer,
  labelConverter))

// El modelo entrena y asi los datos indexados corren.
val model = pipeline.fit(trainingData)

//Muestreo de la prediccion
val predictions = model.transform(testData)
predictions.show(5)

// Se muestra los resultados del modelo de prediccion  y se realiza el test de error
val evaluator = new MulticlassClassificationEvaluator()
.setLabelCol("indexedLabel")
.setPredictionCol("prediction")
.setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//D).Explique detalladamente la funcion matematica de entrenamiento que utilizo con sus propias palabras
/*
Se trata de un clasificador basado en la red neuronal. Esto consta de múltiples capas de nodos.
Al igual que cada capa está completamente conectada a las capas de la red neuronal. Y asi
Los nodos en la capa de entrada representan los datos de entrada.
*/

//E).Explique la funcion de error que utilizo para el resultado final
/*
Esta funcion se basa empleacion propagacion trasera asi para aprender sobre el modelo.
Y usamos la función de la pérdida del nodo para la optimización..
*/
