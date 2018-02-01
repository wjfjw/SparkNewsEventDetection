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

import com.couchbase.client.java.Bucket;
import com.couchbase.client.java.Cluster;
import com.couchbase.client.java.CouchbaseCluster;
import com.couchbase.client.java.document.JsonDocument;
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
import static com.couchbase.spark.japi.CouchbaseDocumentRDD.couchbaseDocumentRDD;

public class App 
{
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static CouchbaseSparkContext csc;
	
	private static Cluster cluster;
	private static Bucket bucket;
	private static final String bucketName = "newsEventDetection";
	
	//算法参数
	private static double singlePassThreshold = 0.2;
	private static int singlePassTimeWindow_hour = 24;		//单位：小时
	
	private static String news_category =  "gn";
	
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
		
		// Create a N1QL Primary Index (but ignore if it exists)
        bucket.bucketManager().createN1qlPrimaryIndex(true, false);
        
        //断开数据库连接
        bucket.close();
        cluster.disconnect();
	}
	

	
	private static void detecteEvent() 
	{
		//从Couchbase中读取由新闻id和content构成的newsRDD
		Statement statement = select("n.news_id", "n.news_time", "n.news_content")
				.from(i(bucketName).as("n"))
				.where( x("news_category").eq(s(news_category)).and( x("news_time").between( s("201711010000").and(s("201711302359")) ) ) )
				.orderBy(Sort.asc("n.news_time"));
		N1qlQuery query = N1qlQuery.simple(statement);
		JavaRDD<CouchbaseQueryRow> newsRDD = csc.couchbaseQuery(query);
		List<CouchbaseQueryRow> resultRowList = newsRDD.collect();
		
		//获取查询结果中每一篇新闻的id, time, content
		List<Integer> idList = new ArrayList<Integer>();
		List<Long> timeList = new ArrayList<Long>();
		List<String> contentList = new ArrayList<String>();
		for(CouchbaseQueryRow row : resultRowList) {
			JsonObject newsObject = row.value();
			idList.add( newsObject.getInt("news_id") );
			timeList.add( newsObject.getLong("news_time") );
			contentList.add( newsObject.getString("news_content") );
		}
		
		JavaRDD<String> contentRDD = sc.parallelize(contentList);
		
//		//新闻id和content构成的newsPairRDD
//		JavaPairRDD<String, String> newsPairRDD = newsRDD.mapToPair( (CouchbaseQueryRow row) -> {
//			JsonObject newsObject = row.value();
//			return new Tuple2<String, String>(newsObject.getString("id"), newsObject.getString("content"));
//		});
//		
//		JavaRDD<String> idRDD = newsPairRDD.keys();
//		JavaRDD<String> contentRDD = newsPairRDD.values();
//		
//		//不知道为什么要这样做才不报错
//		List<String> contentList = contentRDD.collect();
//		JavaRDD<String> contentRDD2 = sc.parallelize(contentList);
		
		//分词
		JavaRDD<List<String>> contentWordsRDD = contentRDD.map( (String content)-> {
			return WordSegmentation.FNLPSegment(content);
		});
	
		//tf-idf特征向量
		JavaRDD<Vector> vectorRDD = FeatureExtraction.getTfidfRDD(2000, contentWordsRDD);
		List<Vector> vectorList = vectorRDD.collect();
		
		singlePass_detecte(idList, timeList, vectorList);
	}
	
	/**
	 * 使用singlePass算法进行事件检测
	 * @param idRDD
	 * @param vectorRDD
	 */
	private static void singlePass_detecte(List<Integer> idList, List<Long> timeList, List<Vector> vectorList)
	{
		//构建featureList
		List<NewsFeature> featureList = new ArrayList<NewsFeature>();
		for(int i=0 ; i<idList.size() ; ++i) {
			featureList.add( new NewsFeature(idList.get(i), timeList.get(i), vectorList.get(i)) );
		}
		
		//singlePass聚类
		List<Event> resultEventList = SinglePassClustering.singlePass(featureList, singlePassThreshold, singlePassTimeWindow_hour);
		
		//根据算法参数查询algorithm_id
		Statement statement = select("n.algorithm_id")
				.from(i(bucketName).as("n"))
				.where( x("algorithm_name").eq(s("single_pass"))
						.and( x("algorithm_parameters.similarity_threshold").eq( s(Double.toString(singlePassThreshold)) ) )
						.and( x("algorithm_parameters.time_window").eq( s(Integer.toString(singlePassTimeWindow_hour)) ) ) );
		N1qlQuery query = N1qlQuery.simple(statement);
		N1qlQueryResult result = bucket.query(query);
		List<N1qlQueryRow> resultRowList = result.allRows();
		if(resultRowList.isEmpty() || !resultRowList.get(0).value().containsKey("algorithm_id")) {
			System.out.println("\n*********************************");
			System.out.println("对应的算法不存在！");
			System.out.println("*********************************\n");
			return;
		}
		int algorithm_id = resultRowList.get(0).value().getInt("algorithm_id");
		
		InsertDataToDB.insertEvent(sc, bucket, resultEventList, algorithm_id, news_category);
	}
	
	

}


////输出singlePass聚类结果
//System.out.println("\n*********************************");
//for(int i=0 ; i<resultClusterList.size() ; ++i) {
//	Cluster cluster = resultClusterList.get(i);
//	System.out.print("[" + (i+1) + ": ");
//	for(NewsFeature feature : cluster.getFeatureList()) {
//		String id = feature.getId();
//		System.out.print(id + ", ");
//	}
//	System.out.println("]");
//}
//System.out.println("*********************************\n");


//System.out.println("\n*********************************");
//System.out.println("Yes");
//System.out.println("*********************************\n");


////特征降维
//featureRDD = FeatureExtraction.getPCARDD(featureRDD, 200);
//
////归一化
//Normalizer normalizer = new Normalizer();
//featureRDD = normalizer.transform(featureRDD);


////KMeans
//int numClusters = 200;
//int numIterations = 20; 
//int runs = 3;
//KMeansModel kMeansModel = KMeans.train(featureRDD.rdd(), numClusters, numIterations, runs);
//JavaRDD<Integer> clusterResultRDD =  kMeansModel.predict( featureRDD );



//输出KMeans聚类结果
//List<Integer> clusterResult = clusterResultRDD.collect();
//Map<Integer, List<Integer>> clusterMap = new HashMap<Integer, List<Integer>>();
//for(int i=0 ; i<clusterResult.size() ; ++i) {
//	int clusterId = clusterResult.get(i);
//	if(clusterMap.containsKey(clusterId)) {
//		clusterMap.get(clusterId).add(i+1);
//	}else {
//		clusterMap.put(clusterId, new ArrayList<Integer>(
//				Arrays.asList(i+1)));
//	}
//}
//System.out.println("\n*********************************");
//for(int clusterId : clusterMap.keySet()) {
//	System.out.print("[" + clusterId + ": ");
//	for(int vectorId : clusterMap.get(clusterId)) {
//		System.out.print(vectorId + ",");
//	}
//	System.out.println("]");
//}
//System.out.println("*********************************\n");


////输出向量之间的相似度
//List<Vector> vectors = featureRDD.collect();
//System.out.println("\n*********************************");
//for(int i=0 ; i<vectors.size() ; ++i) {
//	for(int j=i+1 ; j<vectors.size() ; ++j) {
//		System.out.println( Similarity.getCosineSimilarity(vectors.get(i), vectors.get(j)) );
//	}
//}
//System.out.println("*********************************\n");


////输出分词结果
//System.out.println("\n++++++++++++++++++++++++++++++++");
//for(List<String> list : segmentedLines.collect()) {
//	System.out.println(list);
//}
//System.out.println("++++++++++++++++++++++++++++++++\n");

////输出特征向量
//List<Vector> tfidfVectors = tfidf.collect();
//System.out.println("\n*********************************");
//System.out.println("特征向量");
//for(Vector v : tfidfVectors) {
//	System.out.println(v);
//}
//System.out.println("*********************************\n");

////输出PCA降维后的特征向量
//System.out.println("\n*********************************");
//System.out.println("PCA降维后的特征向量");
//for(Vector v : pcaVectors) {
//	System.out.println(v);
//}
//System.out.println("*********************************\n");

