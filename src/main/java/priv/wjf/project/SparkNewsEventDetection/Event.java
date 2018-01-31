package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class Event 
{
	private List<NewsFeature> featureList;
	private Vector centerVector;
	private long startTime;
	private long endTime;
	
	public Event(Vector v, String id, long startTime) {
		featureList = new ArrayList<NewsFeature>();
		featureList.add( new NewsFeature(id, v) );
		this.centerVector = v;
		this.startTime = startTime;
		this.endTime = Long.MAX_VALUE;
	}

	public long getStartTime() {
		return startTime;
	}
	
	public long getEndTime() {
		return endTime;
	}
	
	public void setEndTime() {
		int size = featureList.size();
		String endId = featureList.get(size-1).getId();
		long endTime = Long.parseLong(endId.substring(0, 12));
		this.endTime = endTime;
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
