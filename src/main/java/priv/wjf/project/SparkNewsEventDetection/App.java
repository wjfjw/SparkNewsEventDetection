package priv.wjf.project.SparkNewsEventDetection;

import static com.couchbase.spark.japi.CouchbaseDocumentRDD.couchbaseDocumentRDD;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

import com.couchbase.client.java.document.JsonDocument;
import com.couchbase.client.java.document.json.JsonObject;

import au.com.bytecode.opencsv.CSVReader;

public class App 
{
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static double singlePassThreshold = 0.2;
	
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
		//新闻格式：id，title,category,url,source,content
		
		List<String> idList = new ArrayList<String>();
		List<String> contentList = new ArrayList<String>();
		List<JsonDocument> jsonDocumentList = new ArrayList<JsonDocument>();
		
		
		//读取CSV文件中的数据
		JavaRDD<String> csvData = sc.textFile("/home/wjf/Data/de-duplicate/201711/category/20171101gn.csv");
		
		//提取新闻的各个属性
		JavaRDD<String[]> newsDataRDD = csvData.map((String line)-> {
			return new CSVReader(new StringReader(line) , ',').readNext();
		});

		//将新闻存储到Couchbase中
		List<String[]> newsData = newsDataRDD.collect();
		for(String[] line : newsData) {
			if(line.length != 6) {
				continue;
			}
			idList.add(line[0]);
			contentList.add(line[5]);
			
			//构建一篇新闻的Json数据
			JsonObject news = JsonObject.create()
	                .put("type", "news")
	                .put("id", line[0])
	                .put("title", line[1])
	                .put("category", line[2])
	                .put("url", line[3])
	                .put("source", line[4])
	                .put("content", line[5])
	                ;
			jsonDocumentList.add( JsonDocument.create("news_"+line[0], news) );
		}
		couchbaseDocumentRDD( sc.parallelize(jsonDocumentList) ).saveToCouchbase();
		
		
		//分词
		JavaRDD<String> contentRDD = sc.parallelize(contentList);
		JavaRDD<List<String>> segmentedRDD = contentRDD.map( (String line)-> {
			return WordSegmentation.FNLPSegment(line);
		});
	
		
		//tf-idf特征向量
		JavaRDD<Vector> featureRDD = FeatureExtraction.getTfidfRDD(2000, segmentedRDD);
		
		
		//singlePass聚类
		List<Cluster> resultClusterList = SinglePassClustering.singlePass(featureRDD, idList, singlePassThreshold);
		
		//输出singlePass聚类结果
		System.out.println("\n*********************************");
		for(int i=0 ; i<resultClusterList.size() ; ++i) {
			Cluster cluster = resultClusterList.get(i);
			System.out.print("[" + (i+1) + ": ");
			for(String id : cluster.getIdList()) {
				System.out.print(id + ", ");
			}
			System.out.println("]");
		}
		System.out.println("*********************************\n");
		
		
		
//		//特征降维
//		featureRDD = FeatureExtraction.getPCARDD(featureRDD, 200);
//		
//		//归一化
//		Normalizer normalizer = new Normalizer();
//		featureRDD = normalizer.transform(featureRDD);

		
//		//KMeans
//		int numClusters = 200;
//	    int numIterations = 20; 
//	    int runs = 3;
//		KMeansModel kMeansModel = KMeans.train(featureRDD.rdd(), numClusters, numIterations, runs);
//		JavaRDD<Integer> clusterResultRDD =  kMeansModel.predict( featureRDD );
		

		
		//输出KMeans聚类结果
//		List<Integer> clusterResult = clusterResultRDD.collect();
//		Map<Integer, List<Integer>> clusterMap = new HashMap<Integer, List<Integer>>();
//		for(int i=0 ; i<clusterResult.size() ; ++i) {
//			int clusterId = clusterResult.get(i);
//			if(clusterMap.containsKey(clusterId)) {
//				clusterMap.get(clusterId).add(i+1);
//			}else {
//				clusterMap.put(clusterId, new ArrayList<Integer>(
//						Arrays.asList(i+1)));
//			}
//		}
//		System.out.println("\n*********************************");
//		for(int clusterId : clusterMap.keySet()) {
//			System.out.print("[" + clusterId + ": ");
//			for(int vectorId : clusterMap.get(clusterId)) {
//				System.out.print(vectorId + ",");
//			}
//			System.out.println("]");
//		}
//		System.out.println("*********************************\n");
		


//		//输出向量之间的相似度
//		List<Vector> vectors = featureRDD.collect();
//		System.out.println("\n*********************************");
//		for(int i=0 ; i<vectors.size() ; ++i) {
//			for(int j=i+1 ; j<vectors.size() ; ++j) {
//				System.out.println( Similarity.getCosineSimilarity(vectors.get(i), vectors.get(j)) );
//			}
//		}
//		System.out.println("*********************************\n");
		
		
		
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
