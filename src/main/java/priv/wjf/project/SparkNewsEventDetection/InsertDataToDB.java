package priv.wjf.project.SparkNewsEventDetection;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import static com.couchbase.spark.japi.CouchbaseDocumentRDD.couchbaseDocumentRDD;

import com.couchbase.client.java.Bucket;
import com.couchbase.client.java.Cluster;
import com.couchbase.client.java.CouchbaseCluster;
import com.couchbase.client.java.document.JsonDocument;
import com.couchbase.client.java.document.json.JsonArray;
import com.couchbase.client.java.document.json.JsonObject;

import au.com.bytecode.opencsv.CSVReader;

public class InsertDataToDB 
{
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static String inputPath = "/home/wjf/Data/de-duplicate/201711/all_summary.csv";
	
	static
	{
		conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.setMaster("local")
				.set("com.couchbase.bucket.newsEventDetection", "");
		
		sc = new JavaSparkContext(conf);
	}

	public static void main(String[] args) 
	{
		// Initialize the Connection
		Cluster cluster = CouchbaseCluster.create("localhost");
		Bucket bucket = cluster.openBucket("newsEventDetection");
		
		//将新闻数据存储到数据库中
		insertData();
		
		// Create a N1QL Primary Index (but ignore if it exists)
        bucket.bucketManager().createN1qlPrimaryIndex(true, false);
        
        //断开数据库连接
        bucket.close();
        cluster.disconnect();
	}
	
	
	/**
	 * 将新闻正文分词，并将新闻数据存储在数据库中
	 */
	private static void insertData()
	{
		List<JsonDocument> jsonDocumentList = new ArrayList<JsonDocument>();
		
		//读取CSV文件中的数据
		JavaRDD<String> csvData = sc.textFile(inputPath);
		
		//提取新闻的各个属性
		JavaRDD<String[]> newsDataRDD = csvData.map((String line)-> {
			return new CSVReader(new StringReader(line) , ',').readNext();
		});

		//将新闻存储到Couchbase中
		List<String[]> newsData = newsDataRDD.collect();
		for(String[] line : newsData) {
			if(line.length != 7) {
				System.out.println("新闻" + line[0] + "csv格式不正确");
				continue;
			}
			
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
	                .put("id", line[0])
	                .put("title", line[1])
	                .put("category", line[2])
	                .put("url", line[3])
	                .put("source", line[4])
	                .put("content", content)
	                .put("summary", line[6])
	                .put("named_entity", nerObject)
	                ;
			jsonDocumentList.add( JsonDocument.create("news_"+line[0], newsObject) );
		}
		
		couchbaseDocumentRDD( sc.parallelize(jsonDocumentList) ).saveToCouchbase();
	}

}
