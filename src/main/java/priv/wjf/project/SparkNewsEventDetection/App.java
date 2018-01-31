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

import com.couchbase.client.java.Bucket;
import com.couchbase.client.java.Cluster;
import com.couchbase.client.java.CouchbaseCluster;
import com.couchbase.client.java.document.json.JsonArray;
import com.couchbase.client.java.document.json.JsonObject;
import com.couchbase.client.java.query.N1qlQuery;
import com.couchbase.client.java.query.N1qlQueryResult;
import com.couchbase.client.java.query.N1qlQueryRow;
import com.couchbase.client.java.query.Statement;
import com.couchbase.client.java.query.dsl.Sort;
import com.couchbase.spark.japi.CouchbaseSparkContext;
import com.couchbase.spark.rdd.CouchbaseQueryRow;

import static com.couchbase.client.java.query.Select.select;
import static com.couchbase.client.java.query.dsl.Expression.i;
import static com.couchbase.client.java.query.dsl.Expression.s;
import static com.couchbase.client.java.query.dsl.Expression.x;

import scala.Tuple2;

public class App 
{
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static CouchbaseSparkContext csc;
	
	private static Cluster cluster;
	private static Bucket bucket;
	private static final String bucketName = "newsEventDetection";
	
	private static double singlePassThreshold = 0.2;
	private static long singlePassTimeWindow_hour = 24;		//单位：小时
	
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
		cluster = CouchbaseCluster.create("localhost");
		bucket = cluster.openBucket(bucketName);
		
		//进行新闻事件检测
		detecteEvent();
        
        //断开数据库连接
        bucket.close();
        cluster.disconnect();
	}
	
	private static void singlePass(JavaRDD<String> idRDD, JavaRDD<Vector> vectorRDD)
	{
		//构建featureList
		List<NewsFeature> featureList = new ArrayList<NewsFeature>();
		List<String> idList = idRDD.collect();
		List<Vector> vectorList = vectorRDD.collect();
		for(int i=0 ; i<idList.size() ; ++i) {
			featureList.add( new NewsFeature(idList.get(i), vectorList.get(i)) );
		}
		
		//singlePass聚类
		List<Event> resultEventList = SinglePassClustering.singlePass(featureList, singlePassThreshold, singlePassTimeWindow_hour);

		//查询最大的id
		int max_event_id = 0;
		Statement statement1 = select("MAX(n.event_id)")
				.from(i(bucketName).as("n"));
		N1qlQuery query1 = N1qlQuery.simple(statement1);
		N1qlQueryResult result1 = bucket.query(query1);
		List<N1qlQueryRow> resultRowList1 = result1.allRows();
		if(!resultRowList1.isEmpty()) {
			max_event_id = resultRowList1.get(0).value().getInt("event_id");
		}
		
		//根据算法参数查询algorithm_id
		Statement statement2 = select("n.algorithm_id")
				.from(i(bucketName).as("n"))
				.where( x("type").eq(s("algorithm"))
						.and( x("algorithm_name").eq(s("singlePass")) )
						.and( x("parameters.similarity_threshold").eq( s(Double.toString(singlePassThreshold)) ) )
						.and( x("parameters.time_window").eq( s(Long.toString(singlePassTimeWindow_hour)) ) ) );
		N1qlQuery query2 = N1qlQuery.simple(statement2);
		N1qlQueryResult result2 = bucket.query(query2);
		List<N1qlQueryRow> resultRowList2 = result2.allRows();
		if(resultRowList2.isEmpty()) {
			System.out.println("对应的算法不存在！");
			return;
		}
		int algorithm_id = resultRowList2.get(0).value().getInt("algorithm_id");
		
		//将event插入数据库中
		int event_id = max_event_id + 1;
		List<JsonObject> eventObjectList = new ArrayList<JsonObject>();
		for(Event event : resultEventList) {
			JsonObject eventObject = JsonObject.create()
					.put("event_id", event_id)
					.put("start_time", event.getStartTime())
					.put("end_time", event.getEndTime())
					.put("news_list", JsonArray.from(event.getFeatureList()))
					.put("algorithm_id", algorithm_id);
			++event_id;
		}
		
		
//		//输出singlePass聚类结果
//		System.out.println("\n*********************************");
//		for(int i=0 ; i<resultClusterList.size() ; ++i) {
//			Cluster cluster = resultClusterList.get(i);
//			System.out.print("[" + (i+1) + ": ");
//			for(NewsFeature feature : cluster.getFeatureList()) {
//				String id = feature.getId();
//				System.out.print(id + ", ");
//			}
//			System.out.println("]");
//		}
//		System.out.println("*********************************\n");
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
		
		//不知道为什么要这样做才不报错
		List<String> contentList = contentRDD.collect();
		JavaRDD<String> contentRDD2 = sc.parallelize(contentList);
		
		//分词
		JavaRDD<List<String>> contentWordsRDD = contentRDD2.map( (String content)-> {
			return WordSegmentation.FNLPSegment(content);
		});
	
		//tf-idf特征向量
		JavaRDD<Vector> vectorRDD = FeatureExtraction.getTfidfRDD(2000, contentWordsRDD);

		singlePass(idRDD, vectorRDD);
		

		
		
//		System.out.println("\n*********************************");
//		System.out.println("Yes");
//		System.out.println("*********************************\n");
		
		
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
