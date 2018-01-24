package priv.wjf.project.SparkNewsEventDetection;

import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.linalg.Vector;

public class Similarity 
{
	/**
	 * 获取两个向量的余弦相似度
	 * @param v1
	 * @param v2
	 * @return v1和v2的余弦相似度
	 */
	public static double getCosineSimilarity(Vector v1, Vector v2) {
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
	public static double getDotProduct(Vector v1, Vector v2) {
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
