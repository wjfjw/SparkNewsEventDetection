package priv.wjf.project.SparkNewsEventDetection;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Row;

import com.couchbase.client.java.Bucket;
import com.couchbase.client.java.CouchbaseCluster;
import com.couchbase.client.java.document.JsonDocument;
import com.couchbase.client.java.document.json.JsonObject;
import com.couchbase.client.java.query.N1qlQuery;
import com.couchbase.client.java.query.Statement;
import com.couchbase.client.java.query.dsl.Sort;
import com.couchbase.client.java.query.dsl.path.AsPath;
import com.couchbase.spark.japi.CouchbaseSparkContext;
import com.couchbase.spark.rdd.CouchbaseQueryRow;

import static com.couchbase.client.java.query.Select.select;
import static com.couchbase.client.java.query.dsl.Expression.i;
import static com.couchbase.client.java.query.dsl.Expression.s;
import static com.couchbase.client.java.query.dsl.Expression.x;

import au.com.bytecode.opencsv.CSVReader;
import scala.Tuple2;

public class App 
{
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static CouchbaseSparkContext csc;
	private static double singlePassThreshold = 0.2;
	private static final String bucketName = "newsEventDetection";
	
	static
	{
		conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.setMaster("local")
				.set("com.couchbase.bucket." + bucketName, "");
		
		sc = new JavaSparkContext(conf);
		csc = CouchbaseSparkContext.couchbaseContext(sc);
	}

	public static void main(String[] args) 
	{
		// Initialize the Connection
		com.couchbase.client.java.Cluster cluster = CouchbaseCluster.create("localhost");
		Bucket bucket = cluster.openBucket(bucketName);
		
		//进行新闻事件检测
		detecteEvent();
        
        //断开数据库连接
        bucket.close();
        cluster.disconnect();
	}
	
	
	private static void detecteEvent() 
	{
		//新闻格式：id，title,category,url,source,content
		
		//从Couchbase中读取由新闻id和content构成的newsRDD
		Statement statement = select("n.id", "n.content")
				.from(i(bucketName).as("n"))
				.where( x("category").eq(s("gn")).and( x("id").between( s("20171101000001").and(s("20171101235999")) ) ) )
				.orderBy(Sort.asc("n.id"));
		N1qlQuery query = N1qlQuery.simple(statement);
		JavaRDD<CouchbaseQueryRow> newsRDD = csc.couchbaseQuery(query);
		
		
		
		//新闻id和content构成的newsPairRDD
		JavaPairRDD<String, String> newsPairRDD = newsRDD.mapToPair( (CouchbaseQueryRow row) -> {
			JsonObject newsObject = row.value();
			return new Tuple2<String, String>(newsObject.getString("id"), newsObject.getString("content"));
		});
		
		JavaRDD<String> idRDD = newsPairRDD.keys();
		JavaRDD<String> contentRDD = newsPairRDD.values();
		
		//分词
		JavaRDD<List<String>> contentWordsRDD = contentRDD.map( (String content)-> {
			return WordSegmentation.FNLPSegment(content);
		});
	
		//tf-idf特征向量
		JavaRDD<Vector> vectorRDD = FeatureExtraction.getTfidfRDD(2000, contentWordsRDD);
		
		//构建featureList
		List<NewsFeature> featureList = new ArrayList<NewsFeature>();
		List<String> idList = idRDD.collect();
		List<Vector> vectorList = vectorRDD.collect();
		for(int i=0 ; i<idList.size() ; ++i) {
			featureList.add( new NewsFeature(idList.get(i), vectorList.get(i)) );
		}
		
		
		
		//singlePass聚类
		List<Cluster> resultClusterList = SinglePassClustering.singlePass(featureList, singlePassThreshold);
		
//		System.out.println("\n*********************************");
//		System.out.println("Yes");
//		System.out.println("*********************************\n");
		
		//输出singlePass聚类结果
		System.out.println("\n*********************************");
		for(int i=0 ; i<resultClusterList.size() ; ++i) {
			Cluster cluster = resultClusterList.get(i);
			System.out.print("[" + (i+1) + ": ");
			for(NewsFeature feature : cluster.getFeatureList()) {
				String id = feature.getId();
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
