package priv.wjf.project.SparkNewsEventDetection;

import java.io.FileWriter;
import java.io.IOException;
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

public class App_experiment 
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
	private static double single_pass_clustering_threshold = 0.6;
	private static int single_pass_time_window = 24;		//单位：小时
	
	//kmeans
	private static int kmeans_cluster_number = 100;
	private static int kmeans_time_window = 24;		//单位：小时
	
	//topic tracking
	private static double topic_tracking_threshold = 0.7;

	public static void main(String[] args) throws IOException 
	{
		//初始化Spark
		conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.set("com.couchbase.bucket." + bucketName, "");
		
		sc = new JavaSparkContext(conf);
		csc = CouchbaseSparkContext.couchbaseContext(sc);
		
		// Initialize the Connection
		cluster = CouchbaseCluster.create("localhost");
		bucket = cluster.openBucket(bucketName);
		
		//进行新闻事件检测
		FileWriter statisticFile = new FileWriter("statistic.csv" , true);
		kmeans_detecte_event(statisticFile);
		
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
	 * @throws IOException 
	 */
	private static void singlePass_detecte_event(FileWriter statisticFile) throws IOException
	{
		long[] t = new long[10];
		t[0] = System.currentTimeMillis();
		
		//进行事件检测的起始时间和结束时间
		long startTime = TimeConversion.getMilliseconds("201711010000");
		long endTime = TimeConversion.getMilliseconds("201711302359");
		
		//查询指定的新闻
		Statement statement = select("news_id", "news_time", "news_content")
				.from(i(bucketName))
				.where( x("news_category").eq(s(news_category)).and( x("news_time").between( x(startTime).and(x(endTime)) ) ) )
				.orderBy(Sort.asc("news_time"));
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
		
		t[1] = System.currentTimeMillis();
		
		//分词
		JavaRDD<List<String>> contentWordsRDD = contentRDD.map( (String content)-> {
			return WordSegmentation.FNLPSegment(content);
		});
		
		t[2] = System.currentTimeMillis();
	
		//tf-idf特征向量
		JavaRDD<Vector> vectorRDD = FeatureExtraction.getTfidfRDD(2500, contentWordsRDD);
		
		t[3] = System.currentTimeMillis();
		
		//特征降维
		vectorRDD = FeatureExtraction.getPCARDD(vectorRDD, 250);
		
		t[4] = System.currentTimeMillis();
		
		List<Vector> vectorList = vectorRDD.collect();
		
		//构建featureList
		List<NewsFeature> featureList = new ArrayList<NewsFeature>();
		for(int i=0 ; i<idList.size() ; ++i) {
			featureList.add( new NewsFeature(idList.get(i), timeList.get(i), vectorList.get(i)) );
		}
		
//		double[] sim_threshold_array = {0.4, 0.5, 0.6, 0.7, 0.8};
		double[] sim_threshold_array = {0.9};
		int[] time_window_array = {24, 36, 48, 60, 72, 96};
//		double sim_threshold = 0.66
//		int time_window = 24;
		double topic_tracking_threshold = 0.7;
		
		for(int time_window : time_window_array) {
			statisticFile.write("\n");
			for(double sim_threshold : sim_threshold_array) {
				t[5] = System.currentTimeMillis();
				//singlePass聚类
				List<Event> resultEventList = SinglePass.singlePassClustering(featureList, sim_threshold, time_window);
				t[6] = System.currentTimeMillis();
				
				//构造eventIdList
				List<Integer> eventIdList = new ArrayList<Integer>(resultEventList.size());
				for(int i=0 ; i<resultEventList.size() ; ++i) {
					eventIdList.add(i);
				}
				
				t[7] = System.currentTimeMillis();
				//话题追踪
				List<Topic> resultTopicList = SinglePass.singlePassTracking(resultEventList, eventIdList, topic_tracking_threshold);
				t[8] = System.currentTimeMillis();
				
				output(statisticFile, "Single-Pass", sim_threshold, 0, time_window, resultEventList.size(), resultTopicList.size(),
						t[1]-t[0], t[2]-t[1], t[3]-t[2], t[4]-t[3], t[6]-t[5], t[8]-t[7]);
			}
		}
	}
	
	
	private static void kmeans_detecte_event(FileWriter statisticFile) throws IOException 
	{
//		int[] time_window_array = {24, 36, 48, 60, 72, 96};
		int[] time_window_array = {24};
//		int[] cluster_num_array = {120, 140, 160, 180, 200};
		int[] cluster_num_array = {120};
		long[] t = new long[10];
		List<Event> resultEventList = new ArrayList<Event>();
		
		for(int time_window : time_window_array) {
			statisticFile.write("\n");
			for(int cluster_num : cluster_num_array) {
				resultEventList.clear();
				
				long query_time=0, wordSegment_time=0, featureExtraction_time=0, pca_time=0, eventDetect_time=0, eventTrack_time=0;
				long inc = time_window * 60 * 60 * 1000;
				long startTime = TimeConversion.getMilliseconds("201711010000");
				long endTime = TimeConversion.getMilliseconds("201711302359");

				while(startTime < endTime) {
					t[0] = System.currentTimeMillis();
					
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
					
					t[1] = System.currentTimeMillis();
					query_time += (t[1]-t[0]);
					
					//分词
					JavaRDD<List<String>> contentWordsRDD = contentRDD.map( (String content)-> {
						return WordSegmentation.FNLPSegment(content);
					});
					
					t[2] = System.currentTimeMillis();
					wordSegment_time += (t[2]-t[1]);
				
					//tf-idf特征向量
					JavaRDD<Vector> vectorRDD = FeatureExtraction.getTfidfRDD(2500, contentWordsRDD);
					
					t[3] = System.currentTimeMillis();
					featureExtraction_time += (t[3]-t[2]);
					
					//特征降维
					vectorRDD = FeatureExtraction.getPCARDD(vectorRDD, 250);
					
					t[4] = System.currentTimeMillis();
					pca_time += (t[4]-t[3]);
					
					//归一化
					Normalizer normalizer = new Normalizer();
					vectorRDD = normalizer.transform(vectorRDD);
					
					List<Vector> vectorList = vectorRDD.collect();
					
					t[5] = System.currentTimeMillis();

					//KMeans
					int numIterations = 30;
					int runs = 1;
					KMeansModel kMeansModel = KMeans.train(vectorRDD.rdd(), cluster_num, numIterations, runs);
					JavaRDD<Integer> clusterResultRDD =  kMeansModel.predict( vectorRDD );
					
					t[6] = System.currentTimeMillis();
					eventDetect_time += (t[6]-t[5]);
					
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
				
				//构造eventIdList
				List<Integer> eventIdList = new ArrayList<Integer>(resultEventList.size());
				for(int i=0 ; i<resultEventList.size() ; ++i) {
					eventIdList.add(i);
				}
				
				//话题追踪
				resultEventList.sort( (Event e1, Event e2) -> {
					return (int)(e1.getStartTime() - e2.getStartTime());
				});
				
				t[7] = System.currentTimeMillis();
				List<Topic> resultTopicList = SinglePass.singlePassTracking(resultEventList, eventIdList, topic_tracking_threshold);
				t[8] = System.currentTimeMillis();
				eventTrack_time += (t[8]-t[7]);
				
				output(statisticFile, "kmeans", 0, cluster_num, time_window, resultEventList.size(), resultTopicList.size(),
						t[1]-t[0], t[2]-t[1], t[3]-t[2], t[4]-t[3], t[6]-t[5], t[8]-t[7]);
			}
		}
		

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
	
	
	private static void output(FileWriter statisticFile, String algorithm, double sim_threshold, int win_cluster_num, int time_window, int event_num, int topic_num,
			long query_time, long wordSegment_time, long featureExtraction_time, long pca_time, long eventDetect_time, long eventTrack_time) 
	{
		try {
			long all_time = query_time + wordSegment_time + featureExtraction_time + pca_time + eventDetect_time + eventTrack_time;
			
			statisticFile.write(algorithm);
			if(algorithm.equals("Single-Pass")) {
				statisticFile.write("," + sim_threshold);
			}else {
				statisticFile.write("," + win_cluster_num);
			}
			statisticFile.write("," + time_window);
			statisticFile.write("," + event_num);
			statisticFile.write("," + topic_num);
			statisticFile.write("," + (double)query_time / 1000);
			statisticFile.write("," + (double)wordSegment_time / 1000);
			statisticFile.write("," + (double)featureExtraction_time / 1000);
			statisticFile.write("," + (double)pca_time / 1000);
			statisticFile.write("," + (double)eventDetect_time / 1000);
			statisticFile.write("," + (double)eventTrack_time / 1000);
			statisticFile.write("," + (double)all_time / 1000);
			statisticFile.write("\n");
			statisticFile.flush();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
