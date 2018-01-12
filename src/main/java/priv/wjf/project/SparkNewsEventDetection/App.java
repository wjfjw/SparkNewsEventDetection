package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

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
		JavaRDD<String> lines = sc.textFile("/home/wjf/Data/de-duplicate/de-duplicated/201711/category/20171101gn.csv");
		JavaRDD<List<String>> segmentedLines = lines.map( (String line)-> {
			line = line.replaceFirst("\\d+,\\d+,.+,\\w{2},.+,.+,", "");
			return WordSegmentation.FNLPSegment(line);
		});
		
		//tf-idf特征向量
		JavaRDD<Vector> featureRDD = FeatureExtraction.getTfidfRDD(2000, segmentedLines);
		
		//特征降维
		featureRDD = FeatureExtraction.getPCARDD(featureRDD, 200);
		
		//归一化
		Normalizer normalizer = new Normalizer();
		featureRDD = normalizer.transform(featureRDD);

		
		//KMeans
		int numClusters = 200;
	    int numIterations = 20; 
	    int runs = 3;
		KMeansModel kMeansModel = KMeans.train(featureRDD.rdd(), numClusters, numIterations, runs);
		JavaRDD<Integer> clusterResultRDD =  kMeansModel.predict( featureRDD );
		
		//输出聚类结果
		List<Integer> clusterResult = clusterResultRDD.collect();
		Map<Integer, List<Integer>> clusterMap = new HashMap<Integer, List<Integer>>();
		for(int i=0 ; i<clusterResult.size() ; ++i) {
			int clusterId = clusterResult.get(i);
			if(clusterMap.containsKey(clusterId)) {
				clusterMap.get(clusterId).add(i+1);
			}else {
				clusterMap.put(clusterId, new ArrayList<Integer>(
						Arrays.asList(i+1)));
			}
		}
		System.out.println("\n*********************************");
		for(int clusterId : clusterMap.keySet()) {
			System.out.print("[" + clusterId + ": ");
			for(int vectorId : clusterMap.get(clusterId)) {
				System.out.print(vectorId + ",");
			}
			System.out.println("]");
		}
		System.out.println("*********************************\n");
		
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
