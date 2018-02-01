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

public class SinglePassClustering 
{
	public static List<Event> singlePass(List<NewsFeature> featureList , double simThreshold, int timeWindow_hour) 
	{
		long timeWindow_millisecond = (long)timeWindow_hour * 60 * 60 * 1000;
		List<Event> resultEventList = new ArrayList<Event>();
		Queue<Event> queue = new LinkedList<Event>();
		Event maxSimEvent = null;
		
		for(NewsFeature feature : featureList) {
			double maxSim = Double.NEGATIVE_INFINITY;
			
			int id = feature.getId();
			long startTime = feature.getTime();
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
				//一天的毫秒数为86400005
				while(!queue.isEmpty()
						&& !withinTimeWindow(queue.peek().getStartTime(), event.getStartTime(), timeWindow_millisecond)) {
						resultEventList.add( queue.poll() );
				}
				queue.add(event);
			}
		}
		
		//将queue中剩余的Event加到结果list中
		while( !queue.isEmpty() ) {
			resultEventList.add( queue.poll() );
		}
		
		//设置每一个Event的结束时间
		for(Event event : resultEventList) {
			event.setEndTime();
		}
		
		return resultEventList;
	}
	
	/**
	 * 
	 * @param startTime 
	 * @param endTime
	 * @param windowTime 时间窗口，毫秒
	 * @return startTime是否在时间窗口内
	 */
	private static boolean withinTimeWindow(long startTime, long endTime, long windowTime) {
		//起始日期
		long year = startTime / 100000000;
		long month = (startTime % 100000000) / 1000000;
		long date = (startTime % 1000000) / 10000;
		long hourOfDay = (startTime % 10000) / 100;
		long minute = startTime % 100;
		Calendar startCalendar = Calendar.getInstance();
		startCalendar.set((int)year, (int)month-1, (int)date, (int)hourOfDay, (int)minute);
		
		//结束日期
		year = endTime / 100000000;
		month = (endTime % 100000000) / 1000000;
		date = (endTime % 1000000) / 10000;
		hourOfDay = (endTime % 10000) / 100;
		minute = endTime % 100;
		Calendar endCalendar = Calendar.getInstance();
		endCalendar.set((int)year, (int)month-1, (int)date, (int)hourOfDay, (int)minute);
		
		long milliseconds = endCalendar.getTimeInMillis() - startCalendar.getTimeInMillis();
		
		if(milliseconds > windowTime) {
			return false;
		}
		return true;
	}
	
}
