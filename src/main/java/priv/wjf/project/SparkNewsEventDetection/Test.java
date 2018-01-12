package priv.wjf.project.SparkNewsEventDetection;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;


public class Test {

	public static void main(String[] args) 
	{
		SparkConf conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.setMaster("local")
				;
		
		JavaSparkContext sc = new JavaSparkContext(conf);
		
		JavaRDD<Integer> rdd = sc.parallelize( Arrays.asList(1, 2, 3) );
		
		List<Integer> list =  rdd.take(1);
		System.out.println("\n*********************************");
		System.out.println(list);
		System.out.println("*********************************\n");
		
		list =  rdd.top(1);
		System.out.println("\n*********************************");
		System.out.println(list);
		System.out.println("*********************************\n");
	}

}
