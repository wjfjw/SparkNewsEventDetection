package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import org.apache.spark.mllib.linalg.Vector;

/**
 * 问题：1.新闻编号、时间等元素
 * 		2.数据库中event的存储
 * 		3.数据库中新闻的存储
 * @author wjf
 *
 */

public class SinglePass 
{
	public static List<Event> singlePassClustering(List<NewsFeature> featureList , double simThreshold, int timeWindow_hour) 
	{
		long timeWindow_millisecond = (long)timeWindow_hour * 60 * 60 * 1000;
		List<Event> resultEventList = new ArrayList<Event>();
		Queue<Event> queue = new LinkedList<Event>();
		Event maxSimEvent = null;
		
		for(NewsFeature feature : featureList) {
			double maxSim = Double.NEGATIVE_INFINITY;
			Vector vector = feature.getVector();
			
			for(Event event : queue) {
				double sim = Similarity.getCosineSimilarity(vector, event.getCenterVector());
				if(sim > maxSim) {
					maxSim = sim;
					maxSimEvent = event;
				}
			}
			
			//如果最大相似度大于simThreshold，则将该新闻加入对应的event
			if(maxSim > simThreshold) {
				maxSimEvent.addFeature(feature);
				maxSimEvent.resetCenterVector();
			}
			//否则，根据该新闻创建一个新的event，并加入到queue中
			else {
				Event event = new Event(feature);
				
				//将queue中超过时间窗口的event移出，并加到resultEventList中
				while(!queue.isEmpty()
						&& (event.getStartTime() - queue.peek().getStartTime() > timeWindow_millisecond)) {
						resultEventList.add( queue.poll() );
				}
				queue.add(event);
			}
		}
		
		//将queue中剩余的Event加到结果list中
		while( !queue.isEmpty() ) {
			resultEventList.add( queue.poll() );
		}
		
		//设置每一个Event的开始和结束时间
		for(Event event : resultEventList) {
			event.setStartAndEndTime();
		}
		
		return resultEventList;
	}
	
	
	public static List<Topic> singlePassTracking(List<Event> eventList, List<Integer> eventIdList, double simThreshold)
	{
		List<Topic> resultTopicList = new ArrayList<Topic>();

		Topic maxSimTopic = null;
		
		for(int i=0 ; i<eventList.size() ; ++i) {
			Event event = eventList.get(i);
			int eventId = eventIdList.get(i);
			
			double maxSim = Double.NEGATIVE_INFINITY;
			Vector vector = event.getCenterVector();
			
			for(Topic topic : resultTopicList) {
				double sim = Similarity.getCosineSimilarity(vector, topic.getCenterVector());
				if(sim > maxSim) {
					maxSim = sim;
					maxSimTopic = topic;
				}
			}
			
			//如果最大相似度大于simThreshold，则将该event加入对应的topic
			if(maxSim > simThreshold) {
				maxSimTopic.addEvent(event);
				maxSimTopic.addEventId(eventId);
				maxSimTopic.resetCenterVector();
			}
			//否则，根据该event创建一个新的topic，并加入到resultTopicList中
			else {
				Topic topic = new Topic(event, eventId);
				resultTopicList.add(topic);
			}
		}
		
		return resultTopicList;
	}
	
}
