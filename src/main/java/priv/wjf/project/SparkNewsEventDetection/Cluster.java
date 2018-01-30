package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class Cluster 
{
	private List<NewsFeature> featureList;
	private Vector centerVector;
	private long time;
	
	public Cluster(Vector v, String id, long time) {
		featureList = new ArrayList<NewsFeature>();
		featureList.add( new NewsFeature(id, v) );
		this.centerVector = v;
		this.time = time;
	}

	public long getTime() {
		return time;
	}
	
	public Vector getCenterVector() {
		return centerVector;
	}
	
	public void addFeature(NewsFeature feature) {
		featureList.add(feature);
	}
	
	public List<NewsFeature> getFeatureList(){
		return featureList;
	}
	
	public void resetCenterVector() {
		int size = centerVector.size();
		double[] sumArray = new double[size];
		for(int i=0 ; i<size ; ++i) {
			sumArray[i] = 0;
		}
		
		for(NewsFeature feature : featureList) {
			Vector v = feature.getVector();
			double[] a = v.toArray();
			for(int i=0 ; i<size ; ++i) {
				sumArray[i] += a[i];
			}
		}
		
		for(int i=0 ; i<size ; ++i) {
			sumArray[i] /= featureList.size();
		}
		
		centerVector = new DenseVector(sumArray).toSparse();
	}
}
