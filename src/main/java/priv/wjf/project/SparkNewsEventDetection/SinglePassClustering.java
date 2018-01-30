package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import org.apache.spark.mllib.linalg.Vector;

/**
 * 问题：1.新闻编号、时间等元素
 * 		2.数据库中cluster的存储
 * 		3.数据库中新闻的存储
 * @author wjf
 *
 */

public class SinglePassClustering 
{
	public static List<Cluster> singlePass(List<NewsFeature> featureList , double simThreshold) 
	{
		List<Cluster> resultClusterList = new ArrayList<Cluster>();
		Queue<Cluster> queue = new LinkedList<Cluster>();
		Cluster maxSimCluster = null;
		
		for(NewsFeature feature : featureList) {
			double maxSim = Double.NEGATIVE_INFINITY;
			Vector vector = feature.getVector();
			String id = feature.getId();
			long time = Long.parseLong(id.substring(0, 12));
			
			for(Cluster cluster : queue) {
				double sim = Similarity.getCosineSimilarity(vector, cluster.getCenterVector());
				if(sim > maxSim) {
					maxSim = sim;
					maxSimCluster = cluster;
				}
			}
			
			//如果最大相似度大于simThreshold，则将该新闻加入对应的cluster
			if(maxSim > simThreshold) {
				maxSimCluster.addFeature(feature);
				maxSimCluster.resetCenterVector();
			}
			//否则，根据该新闻创建一个新的cluster，并加入到queue中
			else {
				Cluster c = new Cluster(vector, id, time);
				
				//将queue中超过时间窗口的cluster移出，并加到resultClusterList中
				//一天的毫秒数为86400005
				while(!queue.isEmpty()
						&& !withinTimeWindow(queue.peek().getTime(), c.getTime(), 86400005)) {
						resultClusterList.add( queue.poll() );
				}
				queue.add(c);
			}
		}
		
		//将queue中剩余的Cluster加到结果list中
		while( !queue.isEmpty() ) {
			resultClusterList.add( queue.poll() );
		}
		return resultClusterList;
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
