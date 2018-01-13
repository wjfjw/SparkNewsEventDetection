package priv.wjf.project.SparkNewsEventDetection;

import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.feature.Normalizer;
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
	public static void singlePass(JavaRDD<Vector> featureRDD, double simThreshold) 
	{
		Queue<Cluster> queue = new LinkedList<Cluster>();
		List<Vector> featureList = featureRDD.collect();
		Cluster maxSimCluster = null;
		double maxSim = Double.NEGATIVE_INFINITY;
		
		for(Vector feature : featureList) {
			for(Cluster cluster : queue) {
				double sim = getCosineSimilarity(feature, cluster.getCenterVector());
				if(sim > maxSim) {
					maxSim = sim;
					maxSimCluster = cluster;
				}
			}
			//如果最大相似度大于simThreshold，则将该新闻加入对应的cluster
			if(maxSim > simThreshold) {
				maxSimCluster.addVector(feature);
				maxSimCluster.resetCenterVector();
			}
			//否则，根据该新闻创建一个新的cluster，并加入到queue中
			else {
				Cluster c = new Cluster(feature, time);
				//将queue中超过时间窗口的cluster移出
				while(!queue.isEmpty()) {
					if( !withinTimeWindow(queue.peek().getTime(), c.getTime()) ) {
						queue.poll();
					}
				}
				queue.add(c);
			}
			
		}
	}
	
	
	private static boolean withinTimeWindow(long startTime, long endTime) {
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
		//一天的毫秒数为86400005
		if(milliseconds > 86400005) {
			return false;
		}
		return true;
	}
	
	
	/**
	 * 获取两个向量的余弦相似度
	 * @param v1
	 * @param v2
	 * @return v1和v2的余弦相似度
	 */
	private static double getCosineSimilarity(Vector v1, Vector v2) {
		Normalizer normalizer = new Normalizer();
		Vector normV1 = normalizer.transform(v1);
		Vector normV2 = normalizer.transform(v2);
		
		return getDotProduct(normV1, normV2);
	}
	
	/**
	 * 获取两个向量的数量积（点积）
	 * @param v1
	 * @param v2
	 * @return v1和v2的数量积（点积）
	 */
	private static double getDotProduct(Vector v1, Vector v2) {
		if(v1.size() != v2.size()) {
			return 0;
		}
		int size = v1.size();
		double sum = 0;
		double[] a1 = v1.toArray();
		double[] a2 = v2.toArray();
		for(int i=0 ; i<size ; ++i) {
			sum += (a1[i]*a2[i]);
		}
		
		return sum;
	}
}
