package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.fnlp.nlp.cn.CNFactory;
import org.fnlp.nlp.cn.CNFactory.Models;
import org.fnlp.util.exception.LoadModelException;

public class NamedEntityRecognition 
{
	private static String modelPath = "/home/wjf/JavaProject/SparkNewsEventDetection/lib/models";
	private static CNFactory factory;
	
	static {
		try {
			// 创建中文处理工厂对象，并使用“models”目录下的模型文件初始化
			factory = CNFactory.getInstance(modelPath);
		} catch (LoadModelException e) {
			e.printStackTrace();
		}
	}
	
	public static Map<String, List<String>> FNLPNer(String content)
	{
		// 使用标注器对包含实体名的句子进行标注，得到结果
     	HashMap<String, String> result = factory.ner(content);
     	
     	Map<String, List<String>> nerResult = new HashMap<String, List<String>>();
     	nerResult.put("place", new ArrayList<String>());
     	nerResult.put("person", new ArrayList<String>());
     	nerResult.put("organization", new ArrayList<String>());
     	nerResult.put("entity", new ArrayList<String>());
     	
     	Map<Integer, String> sortedNer = new TreeMap<Integer, String>();
     	for(String key : result.keySet()) {
     		int index = content.indexOf(key);
     		sortedNer.put(index, key);
     	}
     	
     	for(String entity : sortedNer.values()) {
     		switch(result.get(entity)) 
     		{
     		case "地名":
     			nerResult.get("place").add(entity);
     			break;
     		case "人名":
     			nerResult.get("person").add(entity);
     			break;
     		case "机构名":
     			nerResult.get("organization").add(entity);
     			break;
     		case "实体名":
     			nerResult.get("entity").add(entity);
     			break;
     		default:
     			break;
     		}
     	}
     	
     	return nerResult;
	}

}
