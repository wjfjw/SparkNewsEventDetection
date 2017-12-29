package priv.wjf.project.SparkNewsEventDetection;

import java.io.StringReader;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import static com.couchbase.spark.japi.CouchbaseDocumentRDD.couchbaseDocumentRDD;

import com.couchbase.client.java.Bucket;
import com.couchbase.client.java.Cluster;
import com.couchbase.client.java.CouchbaseCluster;
import com.couchbase.client.java.document.JsonDocument;
import com.couchbase.client.java.document.json.JsonObject;

import au.com.bytecode.opencsv.CSVReader;

public class InsertDataToDB 
{
	private static SparkConf conf;
	private static JavaSparkContext sc;
	private static String prefixPath = "file:///home/wjf/JavaProject/SparkNewsEventDetection/";
	private static String inputFile = prefixPath + "data/deduplicatedNews.csv";
	
	static
	{
		conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.setMaster("local")
				.set("com.couchbase.bucket.newsDataMining", "");
		
		sc = new JavaSparkContext(conf);
	}

	public static void main(String[] args) 
	{
		// Initialize the Connection
		Cluster cluster = CouchbaseCluster.create("localhost");
		Bucket bucket = cluster.openBucket("newsDataMining");
		
		//读入CSV数据后对正文分词，然后写入数据到Couchbase中
		List<String[]> textData = readCSV();
		insertData(textData);
		
		// Create a N1QL Primary Index (but ignore if it exists)
        bucket.bucketManager().createN1qlPrimaryIndex(true, false);
        
        //断开数据库连接
        bucket.close();
        cluster.disconnect();
	}
	
	
	/**
	 * 将新闻正文分词，并将新闻数据存储在数据库中
	 */
	private static void insertData(List<String[]> testData)
	{
		for(String[] rowData : testData){
			if(rowData.length < 6)
				continue;
			if(rowData[2].length()==0 || 
				rowData[4].length()==0 ||
				rowData[5].length()==0)
				continue;
			
			//在Spark下对新闻正文进行分词
			JavaRDD<String> content = sc.parallelize( Arrays.asList(rowData[5]) );
			JavaRDD<String> wordsRDD = content.flatMap( (String str)-> {
				return WordSegmentation.IKAloneSegment(str);
			} );
			List<String> words = wordsRDD.collect();
			
			//构建一篇新闻的Json数据
			//label属性暂时没加上去
			JsonObject article = JsonObject.create()
	                .put("type", "article")
	                .put("id", rowData[0])
	                .put("url", rowData[1])
	                .put("title", rowData[2])
	                .put("source", rowData[3])
	                .put("time", rowData[4])
	                .put("content", rowData[5])
	                .put("wordList", words);
			
			//将新闻数据存储到Couchbase中
			couchbaseDocumentRDD(
				    sc.parallelize( Arrays.asList(JsonDocument.create("article_"+rowData[0], article)) )
				).saveToCouchbase();
		}
	}
	
	
	/**
	 * 读取CSV文件中的数据
	 */
	private static List<String[]> readCSV()
	{
		JavaRDD<String> csvData = sc.textFile(inputFile);
		JavaRDD<String[]> newsData = csvData.flatMap((String textFile)-> {
			return new CSVReader(new StringReader(textFile) , '^').readAll();
		});
		return newsData.collect();
	}

}
