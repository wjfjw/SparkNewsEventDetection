package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
import com.couchbase.client.java.document.json.JsonObject;
import com.couchbase.client.java.query.N1qlQuery;
import com.couchbase.client.java.query.N1qlQueryResult;
import com.couchbase.client.java.query.N1qlQueryRow;
import com.couchbase.client.java.query.Statement;
import com.couchbase.client.java.query.dsl.Expression;
import com.couchbase.client.java.query.dsl.Sort;
import com.couchbase.spark.japi.CouchbaseSparkContext;
import com.couchbase.spark.rdd.CouchbaseQueryRow;

import static com.couchbase.client.java.query.Select.select;
import static com.couchbase.client.java.query.dsl.Expression.i;
import static com.couchbase.client.java.query.dsl.Expression.s;
import static com.couchbase.client.java.query.dsl.Expression.x;

public class App 
{
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static CouchbaseSparkContext csc;
	
	private static Cluster cluster;
	private static Bucket bucket;
	private static final String bucketName = "newsEventDetection";
	
	//新闻类别
	private static String news_category =  "gn";
	
	//算法参数
	//singlePass
	private static double single_pass_clustering_threshold = 0.5;
	private static int single_pass_time_window = 24;		//单位：小时
	
	//kmeans
	private static int kmeans_cluster_number = 100;
	private static int kmeans_time_window = 24;		//单位：小时
	
	//topic tracking
	private static double topic_tracking_threshold = 0.7;
	
	
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
		singlePass_detecte_event();
		
		// Create a N1QL Primary Index (but ignore if it exists)
        bucket.bucketManager().createN1qlPrimaryIndex(true, false);
        
        //断开数据库连接
        bucket.close();
        cluster.disconnect();
	}
	
	
	/**
	 * 使用singlePass算法进行事件检测
	 * @param idRDD
	 * @param vectorRDD
	 */
	private static void singlePass_detecte_event()
	{
		//进行事件检测的起始时间和结束时间
		long startTime = TimeConversion.getMilliseconds("201711010000");
		long endTime = TimeConversion.getMilliseconds("201711302359");
		
		//查询指定的新闻
		Statement statement = select("n.news_id", "n.news_time", "n.news_content")
				.from(i(bucketName).as("n"))
				.where( x("n.news_category").eq(s(news_category)).and( x("n.news_time").between( x(startTime).and(x(endTime)) ) ) )
				.orderBy(Sort.asc("n.news_time"));
		N1qlQuery query = N1qlQuery.simple(statement);
		
//		//不知道为什么会出错，rx.exceptions.MissingBackpressureException
//		JavaRDD<CouchbaseQueryRow> newsRDD = csc.couchbaseQuery(query);
//		List<CouchbaseQueryRow> resultRowList = newsRDD.collect();
		
		N1qlQueryResult result = bucket.query(query);
		List<N1qlQueryRow> resultRowList = result.allRows();
		
		//获取查询结果中每一篇新闻的id, time, content
		List<Integer> idList = new ArrayList<Integer>();
		List<Long> timeList = new ArrayList<Long>();
		List<String> contentList = new ArrayList<String>();
		for(N1qlQueryRow row : resultRowList) {
			JsonObject newsObject = row.value();
			idList.add( newsObject.getInt("news_id") );
			timeList.add( newsObject.getLong("news_time") );
			contentList.add( newsObject.getString("news_content") );
		}
		
		JavaRDD<String> contentRDD = sc.parallelize(contentList);
		
		//分词
		JavaRDD<List<String>> contentWordsRDD = contentRDD.map( (String content)-> {
			return WordSegmentation.FNLPSegment(content);
		});
	
		//tf-idf特征向量
		JavaRDD<Vector> vectorRDD = FeatureExtraction.getTfidfRDD(2500, contentWordsRDD);
		
		//特征降维
		vectorRDD = FeatureExtraction.getPCARDD(vectorRDD, 250);
		
		List<Vector> vectorList = vectorRDD.collect();
		
		//构建featureList
		List<NewsFeature> featureList = new ArrayList<NewsFeature>();
		for(int i=0 ; i<idList.size() ; ++i) {
			featureList.add( new NewsFeature(idList.get(i), timeList.get(i), vectorList.get(i)) );
		}
		
		//singlePass聚类
		List<Event> resultEventList = SinglePass.singlePassClustering(featureList, single_pass_clustering_threshold, single_pass_time_window);
		
		int algorithm_id = getAlgorithm_id("single_pass");
		if(algorithm_id == -1) {
			return;
		}
		//存储到数据库中
		List<Integer> eventIdList = InsertDataToDB.insertEvent(sc, bucket, resultEventList, algorithm_id, news_category);
		
		
		//话题追踪
		List<Topic> resultTopicList = SinglePass.singlePassTracking(resultEventList, eventIdList, topic_tracking_threshold);
		//存储到数据库中
		InsertDataToDB.insertTopic(sc, bucket, resultTopicList, algorithm_id, news_category);
	}
	
	
	private static void kmeans_detecte_event() 
	{
		List<Event> resultEventList = new ArrayList<Event>();
		
		long inc = kmeans_time_window * 60 * 60 * 1000;
		long startTime = TimeConversion.getMilliseconds("201711010000");
		long endTime = TimeConversion.getMilliseconds("201711302359");

		while(startTime < endTime) {
			//查询指定的新闻
			Statement statement = select("news_id", "news_time", "news_content")
					.from(i(bucketName))
					.where( x("news_category").eq(s(news_category)).and( x("news_time").between( x(startTime).and(x(startTime + inc)) ) ) );
			N1qlQuery query = N1qlQuery.simple(statement);
			
			startTime += (inc + 1);
			
			N1qlQueryResult result = bucket.query(query);
			List<N1qlQueryRow> resultRowList = result.allRows();
			
			//获取查询结果中每一篇新闻的id, time, content
			List<Integer> idList = new ArrayList<Integer>();
			List<Long> timeList = new ArrayList<Long>();
			List<String> contentList = new ArrayList<String>();
			for(N1qlQueryRow row : resultRowList) {
				JsonObject newsObject = row.value();
				idList.add( newsObject.getInt("news_id") );
				timeList.add( newsObject.getLong("news_time") );
				contentList.add( newsObject.getString("news_content") );
			}
			
			JavaRDD<String> contentRDD = sc.parallelize(contentList);
			
			//分词
			JavaRDD<List<String>> contentWordsRDD = contentRDD.map( (String content)-> {
				return WordSegmentation.FNLPSegment(content);
			});
		
			//tf-idf特征向量
			JavaRDD<Vector> vectorRDD = FeatureExtraction.getTfidfRDD(2500, contentWordsRDD);
			
			//特征降维
			vectorRDD = FeatureExtraction.getPCARDD(vectorRDD, 250);
			
			//归一化
			Normalizer normalizer = new Normalizer();
			vectorRDD = normalizer.transform(vectorRDD);
			
			List<Vector> vectorList = vectorRDD.collect();

			//KMeans
			int numIterations = 30;
			int runs = 3;
			KMeansModel kMeansModel = KMeans.train(vectorRDD.rdd(), kmeans_cluster_number, numIterations, runs);
			JavaRDD<Integer> clusterResultRDD =  kMeansModel.predict( vectorRDD );
			
			List<Integer> clusterResult = clusterResultRDD.collect();
			Map<Integer, Event> map = new HashMap<Integer, Event>();
			for(int i=0 ; i<clusterResult.size() ; ++i) {
				int clusterId = clusterResult.get(i);
				NewsFeature feature = new NewsFeature(idList.get(i), timeList.get(i), vectorList.get(i));
				if(map.containsKey(clusterId)) {
					map.get(clusterId).addFeature(feature);
				}else {
					map.put(clusterId, new Event(feature));
				}
			}
			
			for(Event event : map.values()) {
				event.resetCenterVector();
				event.setStartAndEndTime();
				resultEventList.add(event);
			}
		}
		
		int algorithm_id = getAlgorithm_id("kmeans");
		if(algorithm_id == -1) {
			return;
		}
		//存储到数据库中
		List<Integer> eventIdList = InsertDataToDB.insertEvent(sc, bucket, resultEventList, algorithm_id, news_category);
		
		//话题追踪
		resultEventList.sort( (Event e1, Event e2) -> {
			return (int)(e1.getStartTime() - e2.getStartTime());
		});
		List<Topic> resultTopicList = SinglePass.singlePassTracking(resultEventList, eventIdList, topic_tracking_threshold);
		//存储到数据库中
		InsertDataToDB.insertTopic(sc, bucket, resultTopicList, algorithm_id, news_category);
	}
	
	
	/**
	 * 获取指定算法及参数的id
	 * @param algorithm_name
	 * @return
	 */
	private static int getAlgorithm_id(String algorithm_name)
	{
		Expression expression = x("algorithm_name").eq(s(algorithm_name));
		if(algorithm_name.equals("single_pass")) {
			expression = expression.and( x("algorithm_parameters.similarity_threshold").between( 
					x(single_pass_clustering_threshold-0.01).and(x(single_pass_clustering_threshold+0.01)) ))
					.and( x("algorithm_parameters.time_window").eq( x(single_pass_time_window) ) )
					.and( x("algorithm_parameters.topic_tracking_threshold").between( 
							x(topic_tracking_threshold-0.01).and(x(topic_tracking_threshold+0.01)) ) );
		}else if(algorithm_name.equals("kmeans")) {
			expression = expression.and( x("algorithm_parameters.cluster_number").eq( x(kmeans_cluster_number) ) )
					.and( x("algorithm_parameters.time_window").eq( x(kmeans_time_window) ) )
					.and( x("algorithm_parameters.topic_tracking_threshold").between( 
							x(topic_tracking_threshold-0.01).and(x(topic_tracking_threshold+0.01)) ) );
		}
		
		//根据算法参数查询algorithm_id
		Statement statement = select("algorithm_id")
				.from(i(bucketName))
				.where(expression);

		N1qlQuery query = N1qlQuery.simple(statement);
		N1qlQueryResult result = bucket.query(query);
		List<N1qlQueryRow> resultRowList = result.allRows();
		
		int algorithm_id = -1;
		if(!resultRowList.isEmpty() && resultRowList.get(0).value().containsKey("algorithm_id")) {
			JsonObject object = resultRowList.get(0).value();
			algorithm_id = object.getInt("algorithm_id");
		}else {
			System.out.println("\n*********************************");
			System.out.println("对应的算法不存在！");
			System.out.println("*********************************\n");
		}
		return algorithm_id;
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


////新闻id和content构成的newsPairRDD
//JavaPairRDD<String, String> newsPairRDD = newsRDD.mapToPair( (CouchbaseQueryRow row) -> {
//	JsonObject newsObject = row.value();
//	return new Tuple2<String, String>(newsObject.getString("id"), newsObject.getString("content"));
//});
//
//JavaRDD<String> idRDD = newsPairRDD.keys();
//JavaRDD<String> contentRDD = newsPairRDD.values();
//
////不知道为什么要这样做才不报错
//List<String> contentList = contentRDD.collect();
//JavaRDD<String> contentRDD2 = sc.parallelize(contentList);

