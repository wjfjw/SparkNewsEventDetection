package priv.wjf.project.SparkNewsEventDetection;

import java.io.IOException;
import java.io.StringReader;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.hdfs.server.datanode.tail_jsp;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import au.com.bytecode.opencsv.CSVReader;


public class Test {

	public static void main(String[] args) throws IOException 
	{
//		SparkConf conf = new SparkConf()
//				.setAppName("SparkNewsEventDetection")
//				.setMaster("local")
//				;
//		
//		JavaSparkContext sc = new JavaSparkContext(conf);
//		
//		System.out.println("\n*********************************");
//		System.out.println();
//		System.out.println("*********************************\n");
		
		Scanner input = new Scanner(Paths.get("/home/wjf/Data/de-duplicate/201711/all.csv"));
		int count = 0;
		
		while(input.hasNextLine()) {
			String line = input.nextLine();
			String[] lineArray = new CSVReader(new StringReader(line) , ',').readNext();
			if(lineArray.length != 6) {
				System.out.println(lineArray[0] + '\t' + lineArray.length);
				++count;
			}
		}
		System.out.println(count);
		
		input.close();
		
//		String content = "，近期，波兰宣布将修建一条打通维斯图拉河沙嘴的运河，以便让船只直接进入波罗的海，该项目将于2018年年底开始动工，耗资大约2.1亿欧元。目前，从波兰埃尔布隆格港口驶往波罗的海的船只必须穿过一个狭窄的半岛，并向东驶入俄罗斯加里宁格勒州的海域，才能往北进入大海。波兰海洋经济部长马雷克·格罗巴尔奇克在与地方官员会晤时说：“修建这条运河是政府的首要任务之一。”现在，欧盟与俄罗斯的紧张关系达到几十年来的最高点。近日，莫斯科在其欧洲的飞地加里宁格勒州展开了大规模军事演习。,，近期，波兰宣布将修建一条打通维斯图拉河沙嘴的运河，以便让船只直接进入波罗的海，该项目将于2018年年底开始动工，耗资大约2.1亿欧元。目前，从波兰埃尔布隆格港口驶往波罗的海的船只必须穿过一个狭窄的半岛，并向东驶入俄罗斯加里宁格勒州的海域，才能往北进入大海。波兰海洋经济部长马雷克·格罗巴尔奇克在与地方官员会晤时说：“修建这条运河是政府的首要任务之一。”现在，欧盟与俄罗斯的紧张关系达到几十年来的最高点。近日，莫斯科在其欧洲的飞地加里宁格勒州展开了大规模军事演习。";
//		Map<String, List<String>> map = NamedEntityRecognition.FNLPNer(content);
//		System.out.println(map);
	}

}
