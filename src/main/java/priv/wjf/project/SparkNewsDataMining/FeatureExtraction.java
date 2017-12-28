package priv.wjf.project.SparkNewsDataMining;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vector;
//import org.apache.spark.ml.feature.HashingTF;
//import org.apache.spark.ml.feature.IDF;
//import org.apache.spark.ml.feature.IDFModel;
//import org.apache.spark.ml.feature.Tokenizer;
//import org.apache.spark.sql.DataFrame;
//import org.apache.spark.sql.Row;
//import org.apache.spark.sql.RowFactory;
//import org.apache.spark.sql.SQLContext;
//import org.apache.spark.sql.types.*;

public class FeatureExtraction 
{
	
	private static SparkConf conf;
	private static JavaSparkContext sc;
//	private static SQLContext sqlContext;
	
	static
	{
		conf = new SparkConf()
				.setAppName("SparkNewsDataMining")
				.setMaster("local")
				;
		
		sc = new JavaSparkContext(conf);
//		sqlContext = new SQLContext(sc);
	}

	public static void main(String[] args) 
	{
		JavaRDD<String> lines = sc.parallelize( Arrays.asList(
				"在实践中形成了以新发展理念为主要内容的习近平新时代中国特色社会主义经济思想",
				"在习近平新时代中国特色社会主义经济思想引领下，中国经济发展进入了新时代，由高速增长阶段转向高质量发展阶段",
				"2018考研数学出现 “神押题”，考生怀疑发生泄题。"
				) );
		
		JavaRDD<List<String>> segmentedLines = lines.map( (String line)-> {
			return WordSegmentation.FNLPSegment(line);
		});
		
		System.out.println("\n++++++++++++++++++++++++++++++++");
		for(List<String> list : segmentedLines.collect()) {
			System.out.println(list);
		}
		System.out.println("++++++++++++++++++++++++++++++++\n");
		
		//tf
		HashingTF hashingTF = new HashingTF(20);
		JavaRDD<Vector> tf =  hashingTF.transform(segmentedLines);
		tf.cache();
		
		//idf
		IDF idf = new IDF();
		IDFModel idfModel = idf.fit(tf);
		JavaRDD<Vector> tfidf =  idfModel.transform(tf);
		
		List<Vector> result = tfidf.collect();
		System.out.println("\n*********************************");
		for(Vector v : result) {
			System.out.println(v);
		}
		System.out.println("*********************************\n");
		
//		JavaRDD<Row> jrdd = sc.parallelize(Arrays.asList(
//				  RowFactory.create(0.0, "Hi I heard about Spark"),
//				  RowFactory.create(0.0, "I wish Java could use case classes"),
//				  RowFactory.create(1.0, "Logistic regression models are neat")
//				));
//		
//				StructType schema = new StructType(new StructField[]{
//				  new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
//				  new StructField("sentence", DataTypes.StringType, false, Metadata.empty())
//				});
//				
//				DataFrame sentenceData = sqlContext.createDataFrame(jrdd, schema);
//				
//				Tokenizer tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words");
//				DataFrame wordsData = tokenizer.transform(sentenceData);
//				
//				//TF
//				int numFeatures = 20;
//				HashingTF hashingTF = new HashingTF()
//				  .setInputCol("words")
//				  .setOutputCol("rawFeatures")
//				  .setNumFeatures(numFeatures);
//				DataFrame featurizedData = hashingTF.transform(wordsData);
//				
//				//IDF
//				IDF idf = new IDF().setInputCol("rawFeatures").setOutputCol("features");
//				IDFModel idfModel = idf.fit(featurizedData);
//				DataFrame rescaledData = idfModel.transform(featurizedData);
//				
//				for (Row r : rescaledData.select("features", "label").take(3)) {
//				  Vector features = r.getAs(0);
//				  Double label = r.getDouble(1);
//				  System.out.println(features);
//				  System.out.println(label);
//				}
	}

}
