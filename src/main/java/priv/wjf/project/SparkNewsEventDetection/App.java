package priv.wjf.project.SparkNewsEventDetection;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;

public class App 
{
	private static SparkConf conf;
	private static JavaSparkContext sc;
	
	static
	{
		conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.setMaster("local")
				;
		
		sc = new JavaSparkContext(conf);
	}

	public static void main(String[] args) 
	{
		//分词
		JavaRDD<String> lines = sc.parallelize( Arrays.asList(
				"在实践中形成了以新发展理念为主要内容的习近平新时代中国特色社会主义经济思想",
				"在习近平新时代中国特色社会主义经济思想引领下，中国经济发展进入了新时代，由高速增长阶段转向高质量发展阶段",
				"2018考研数学出现 “神押题”，考生怀疑发生泄题。"
				) );
		JavaRDD<List<String>> segmentedLines = lines.map( (String line)-> {
			return WordSegmentation.FNLPSegment(line);
		});
		
		JavaRDD<Vector> tfidfRDD = FeatureExtraction.getTfidfRDD(500, segmentedLines);
		
		//KMeans
		int numClusters = 10;
	    int numIterations = 30; 
		KMeansModel kMeansModel = KMeans.train(tfidfRDD.rdd(), numClusters, numIterations);
		
		
//		//输出分词结果
//		System.out.println("\n++++++++++++++++++++++++++++++++");
//		for(List<String> list : segmentedLines.collect()) {
//			System.out.println(list);
//		}
//		System.out.println("++++++++++++++++++++++++++++++++\n");
		
//		//输出特征向量
//		List<Vector> tfidfVectors = tfidf.collect();
//		System.out.println("\n*********************************");
//		System.out.println("特征向量");
//		for(Vector v : tfidfVectors) {
//			System.out.println(v);
//		}
//		System.out.println("*********************************\n");
		
//		//输出PCA降维后的特征向量
//		System.out.println("\n*********************************");
//		System.out.println("PCA降维后的特征向量");
//		for(Vector v : pcaVectors) {
//			System.out.println(v);
//		}
//		System.out.println("*********************************\n");

	}

}
