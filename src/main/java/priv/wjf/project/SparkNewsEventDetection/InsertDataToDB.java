package priv.wjf.project.SparkNewsEventDetection;

import java.io.StringReader;
import java.util.ArrayList;
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
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static String inputPath_news = "/home/wjf/Data/de-duplicate/201711/all_summary.csv";
	
	private static Cluster cluster;
	private static Bucket bucket;
	private static final String bucketName = "newsEventDetection";
	
	static
	{
		conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.setMaster("local")
				.set("com.couchbase.bucket." + bucketName, "");
		
		sc = new JavaSparkContext(conf);
	}

	public static void main(String[] args) 
	{
		// Initialize the Connection
		cluster = CouchbaseCluster.create("localhost");
		bucket = cluster.openBucket(bucketName);
		
		//将新闻数据存储到数据库中
		insertNews();
		
		// Create a N1QL Primary Index (but ignore if it exists)
        bucket.bucketManager().createN1qlPrimaryIndex(true, false);
        
        //断开数据库连接
        bucket.close();
        cluster.disconnect();
	}

	
	
	/**
	 * 将新闻数据存储在数据库中
	 */
	public static void insertNews()
	{
		//新闻格式：id，title,category,url,source,content
		List<JsonDocument> jsonDocumentList = new ArrayList<JsonDocument>();
		
		//读取CSV文件中的数据
		JavaRDD<String> csvData = sc.textFile(inputPath_news);
		
		//提取新闻的各个属性
		JavaRDD<String[]> newsDataRDD = csvData.map((String line)-> {
			return new CSVReader(new StringReader(line) , ',').readNext();
		});
		List<String[]> newsData = newsDataRDD.collect();
		
		//获取当前可用的news_id
		int news_id = getMaxId("news_id") + 1;
		
		//将新闻存储到Couchbase中
		for(String[] line : newsData) {
			if(line.length != 7) {
				System.out.println("新闻" + line[0] + "csv格式不正确");
				continue;
			}
			
			long time = Long.parseLong(line[0].substring(0, 12));
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
	 * 将事件存储到数据库中
	 * @param resultEventList
	 * @param algorithm_id
	 */
	public static void insertEvent(List<Event> resultEventList, int algorithm_id, String event_category) 
	{
		//获取当前可用的event_id
		int event_id = getMaxId("event_id") + 1;
		
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
					.put("event_id", event_id)
					.put("event_start_time", event.getStartTime())
					.put("event_end_time", event.getEndTime())
					.put("event_news_list", newsIdArray)
					.put("event_algorithm", algorithm_id)
					.put("event_category", event_category);
			
			jsonDocumentList.add( JsonDocument.create("event_" + event_id, eventObject) );
			++event_id;
		}
		couchbaseDocumentRDD( sc.parallelize(jsonDocumentList) ).saveToCouchbase();
	}
	
	/**
	 * 将算法存储到数据库中
	 */
	private static void insertAlgorithm()
	{
		String[] algorithm_name = {"singlePass", "kmeans"};
		double[] singlePass_similarity_threshold = {0.1, 0.2, 0.3, 0.4, 0.5};
		int[] singlePass_time_window = {12, 24, 36, 48, 60, 72, 84, 96};
		
		int[] kmeans_cluster_number = {50, 100, 150, 200, 250, 300, 350, 400, 450, 500};
		
		//获取当前可用的algorithm_id
		int algorithm_id = getMaxId("algorithm_id") + 1;
		
		List<JsonDocument> jsonDocumentList = new ArrayList<JsonDocument>();
		for(String algorithmName : algorithm_name) {
			JsonObject parametersObject = null;
			if(algorithmName.equals("singlePass")) {
				for(double similarity_threshold : singlePass_similarity_threshold) {
					for(int time_window : singlePass_time_window) {
						//构建parameters的JsonObject
						parametersObject = JsonObject.create()
								.put("similarity_threshold", similarity_threshold)
								.put("time_window", time_window);
					}
				}
			}else {
				for(int cluster_number : kmeans_cluster_number) {
					//构建parameters的JsonObject
					parametersObject = JsonObject.create()
							.put("cluster_number", cluster_number);
				}
			}
			
			//构建algorithm的JsonObject
			JsonObject algorithmObject = JsonObject.create()
	                .put("type", "algorithm")
	                .put("algorithm_id", algorithm_id)
	                .put("algorithm_name", algorithmName)
	                .put("algorithm_parameters", parametersObject);
			
			jsonDocumentList.add( JsonDocument.create("algorithm_"+algorithm_id, algorithmObject) );
			++algorithm_id;
		}
		couchbaseDocumentRDD( sc.parallelize(jsonDocumentList) ).saveToCouchbase();
	}
	
	
	/**
	 * 查询最大的id
	 * @param documentId
	 * @return
	 */
	private static int getMaxId(String documentId)
	{
		int max_id = 0;
		Statement statement = select("MAX(n." + documentId + ")")
				.from(i(bucketName).as("n"));
		N1qlQuery query = N1qlQuery.simple(statement);
		N1qlQueryResult result = bucket.query(query);
		List<N1qlQueryRow> resultRowList = result.allRows();
		if(!resultRowList.isEmpty()) {
			max_id = resultRowList.get(0).value().getInt(documentId);
		}
		return max_id;
	}

}
