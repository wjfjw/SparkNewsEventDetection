package priv.wjf.project.SparkNewsEventDetection;

import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.linalg.Vector;

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
				Cluster frontCluster = queue.peek();
				long interval = 
				if(frontCluster.getTime())
				queue.add(c);
			}
			
		}
	}
	
	
	private static long getInterval(long start, long end) {
		long startDay = start / 10000;
		long endDay = end / 10000;
		long diffDays = endDay - startDay;
		
		long year = start / 100000000;
		long month = (start % 100000000) / 1000000;
		long day = (start % 1000000) / 10000;
		
		Calendar startTime = Calendar.getInstance();
		startTime.set(year, month, date, hourOfDay, minute);
	}
	
	/**时间例子
	 *  201711010000
		201711012359
		201711020000
		201711022359
	 */
	
	
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
