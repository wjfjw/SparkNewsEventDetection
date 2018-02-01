package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

public class Event 
{
	private List<NewsFeature> featureList;
	private Vector centerVector;
	private long startTime;
	private long endTime;
	
	public Event(NewsFeature feature) {
		featureList = new ArrayList<NewsFeature>();
		featureList.add(feature);
		this.centerVector = feature.getVector();
		this.startTime = feature.getTime();
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
		long endTime = featureList.get(size-1).getTime();
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
