package priv.wjf.project.SparkNewsEventDetection;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import static com.couchbase.client.java.query.Select.select;
import static com.couchbase.client.java.query.dsl.Expression.i;
import static com.couchbase.spark.japi.CouchbaseDocumentRDD.couchbaseDocumentRDD;

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

import au.com.bytecode.opencsv.CSVReader;

public class InsertDataToDB 
{
	public static void main(String[] args) 
	{
		final String bucketName = "newsEventDetection";
		SparkConf conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.setMaster("local")
				.set("com.couchbase.bucket." + bucketName, "");
		JavaSparkContext sc = new JavaSparkContext(conf);;

		// Initialize the Connection
		Cluster cluster = CouchbaseCluster.create("localhost");
		Bucket bucket = cluster.openBucket(bucketName);
		
		//将新闻数据存储到数据库中
		insertAlgorithm(sc, bucket);
		
		// Create a N1QL Primary Index (but ignore if it exists)
        bucket.bucketManager().createN1qlPrimaryIndex(true, false);
        
        //断开数据库连接
        bucket.close();
        cluster.disconnect();
	}

	
	
	/**
	 * 将新闻数据存储在数据库中
	 */
	public static void insertNews(JavaSparkContext sc, Bucket bucket)
	{
		//新闻格式：id，title,category,url,source,content
		String inputPath_news = "/home/wjf/Data/de-duplicate/201711/all_summary.csv";
		List<JsonDocument> jsonDocumentList = new ArrayList<JsonDocument>();
		
		//读取CSV文件中的数据
		JavaRDD<String> csvData = sc.textFile(inputPath_news);
		
		//提取新闻的各个属性
		JavaRDD<String[]> newsDataRDD = csvData.map((String line)-> {
			return new CSVReader(new StringReader(line) , ',').readNext();
		});
		List<String[]> newsData = newsDataRDD.collect();
		
		//获取当前可用的news_id
		int news_id = getMaxId(bucket, "news_id") + 1;
		
		//将新闻存储到Couchbase中
		for(String[] line : newsData) {
			if(line.length != 7) {
				System.out.println("新闻" + line[0] + "csv格式不正确");
				continue;
			}
			
			long time = TimeConversion.getMilliseconds(line[0].substring(0, 12));
			String content = line[5];
			
			//获取命名实体
			JsonObject nerObject = JsonObject.create();
			Map<String, List<String>> nerMap = NamedEntityRecognition.FNLPNer(content);
			for(String key : nerMap.keySet()) {
				//每个nerArray为每一类实体
				JsonArray nerArray = JsonArray.create();
				for(String entity : nerMap.get(key)) {
					//过滤一个字符的实体
					if(entity.length() > 1) {
						nerArray.add(entity);
					}
				}
				nerObject.put(key, nerArray);
			}
			
			//构建一篇新闻的Json数据
			JsonObject newsObject = JsonObject.create()
	                .put("type", "news")
	                .put("news_id", news_id)
	                .put("news_time", time)
	                .put("news_title", line[1])
	                .put("news_category", line[2])
	                .put("news_url", line[3])
	                .put("news_source", line[4])
	                .put("news_content", content)
	                .put("news_summary", line[6])
	                .put("news_named_entity", nerObject)
	                ;
			jsonDocumentList.add( JsonDocument.create("news_"+news_id, newsObject) );
			++news_id;
		}
		
		couchbaseDocumentRDD( sc.parallelize(jsonDocumentList) ).saveToCouchbase();
	}
	
	
	/**
	 * 将算法存储到数据库中
	 */
	private static void insertAlgorithm(JavaSparkContext sc, Bucket bucket)
	{
		double[] singlePass_similarity_threshold = {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8};
		int[] singlePass_time_window = {12, 24, 36, 48, 60, 72, 84, 96};
		
		int[] kmeans_cluster_number = {50, 100, 150, 200, 250, 300, 350, 400, 450, 500};
		int[] kmeans_time_window = {12, 24, 36, 48, 60, 72, 84, 96};
		
		double[] topic_tracking_threshold_array = {0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6};
		
		//获取当前可用的algorithm_id
		int algorithm_id = getMaxId(bucket, "algorithm_id") + 1;
		
		List<JsonDocument> jsonDocumentList = new ArrayList<JsonDocument>();
		
		for(double similarity_threshold : singlePass_similarity_threshold) {
			for(int time_window : singlePass_time_window) {
				for(double topic_tracking_threshold : topic_tracking_threshold_array){
					//构建parameters的JsonObject
					JsonObject parametersObject = JsonObject.create()
							.put("similarity_threshold", similarity_threshold)
							.put("time_window", time_window)
							.put("topic_tracking_threshold", topic_tracking_threshold);
					
					//构建algorithm的JsonObject
					JsonObject algorithmObject = JsonObject.create()
			                .put("type", "algorithm")
			                .put("algorithm_id", algorithm_id)
			                .put("algorithm_name", "single_pass")
			                .put("algorithm_parameters", parametersObject);
					
					jsonDocumentList.add( JsonDocument.create("algorithm_"+algorithm_id, algorithmObject) );
					++algorithm_id;
				}
			}
		}
		
		for(int cluster_number : kmeans_cluster_number) {
			for(int time_window : kmeans_time_window) {
				for(double topic_tracking_threshold : topic_tracking_threshold_array){
					//构建parameters的JsonObject
					JsonObject parametersObject = JsonObject.create()
							.put("cluster_number", cluster_number)
							.put("time_window", time_window)
							.put("topic_tracking_threshold", topic_tracking_threshold);
					
					//构建algorithm的JsonObject
					JsonObject algorithmObject = JsonObject.create()
			                .put("type", "algorithm")
			                .put("algorithm_id", algorithm_id)
			                .put("algorithm_name", "kmeans")
			                .put("algorithm_parameters", parametersObject);
					
					jsonDocumentList.add( JsonDocument.create("algorithm_"+algorithm_id, algorithmObject) );
					++algorithm_id;
				}
			}
		}
		couchbaseDocumentRDD( sc.parallelize(jsonDocumentList) ).saveToCouchbase();
	}
	
	
	/**
	 * 将事件存储到数据库中
	 * @param resultEventList
	 * @param algorithm_id
	 */
	public static List<Integer> insertEvent(
			JavaSparkContext sc, Bucket bucket, List<Event> resultEventList, int algorithm_id, String event_category) 
	{
		//获取当前可用的event_id
		int event_id = getMaxId(bucket, "event_id") + 1;
		
		List<Integer> eventIdList = new ArrayList<Integer>();
		
		//将event存储到数据库中
		List<JsonDocument> jsonDocumentList = new ArrayList<JsonDocument>();
		
		for(Event event : resultEventList) {
			//构建事件的新闻id的JsonArray
			JsonArray newsIdArray = JsonArray.create();
			for(NewsFeature feature : event.getFeatureList()) {
				newsIdArray.add(feature.getId());
			}
			//构建事件的JsonObject
			JsonObject eventObject = JsonObject.create()
					.put("type", "event")
					.put("event_id", event_id)
					.put("event_start_time", event.getStartTime())
					.put("event_end_time", event.getEndTime())
					.put("event_news_list", newsIdArray)
					.put("event_algorithm", algorithm_id);
			
			jsonDocumentList.add( JsonDocument.create("event_" + event_id, eventObject) );
			eventIdList.add(event_id);
			++event_id;
		}
		couchbaseDocumentRDD( sc.parallelize(jsonDocumentList) ).saveToCouchbase();
		
		return eventIdList;
	}
	
	
	public static void insertTopic(
			JavaSparkContext sc, Bucket bucket, List<Topic> resultTopicList, int algorithm_id, String topic_category) 
	{
		//获取当前可用的topic_id
		int topic_id = getMaxId(bucket, "topic_id") + 1;
		
		//将topic存储到数据库中
		List<JsonDocument> jsonDocumentList = new ArrayList<JsonDocument>();
		
		for(Topic topic : resultTopicList) {
			//构建topic的event_id的JsonArray
			JsonArray eventIdArray = JsonArray.create();
			for(int event_id : topic.getEventIdList()) {
				eventIdArray.add(event_id);
			}
			//构建topic的JsonObject
			JsonObject topicObject = JsonObject.create()
					.put("type", "topic")
					.put("topic_id", topic_id)
					.put("topic_event_list", eventIdArray)
					.put("topic_algorithm", algorithm_id)
					.put("topic_category", topic_category);
			
			jsonDocumentList.add( JsonDocument.create("topic_" + topic_id, topicObject) );
			++topic_id;
		}
		couchbaseDocumentRDD( sc.parallelize(jsonDocumentList) ).saveToCouchbase();
	}
	
	
	/**
	 * 查询最大的id
	 * @param documentId
	 * @return
	 */
	private static int getMaxId(Bucket bucket, String documentId)
	{
		int max_id = 0;
		Statement statement = select("MAX(n." + documentId + ")")
				.from(i(bucket.name()).as("n"));
		N1qlQuery query = N1qlQuery.simple(statement);
		N1qlQueryResult result = bucket.query(query);
		List<N1qlQueryRow> resultRowList = result.allRows();
		if(!resultRowList.isEmpty()) {
			JsonObject firstObject = resultRowList.get(0).value();
			if(firstObject.containsKey(documentId)) {
				max_id = firstObject.getInt(documentId);
			}
		}
		return max_id;
	}
	
}
