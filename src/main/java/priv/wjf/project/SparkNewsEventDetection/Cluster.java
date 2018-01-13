package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class Cluster 
{
	private List<Vector> vectorList;
	private Vector centerVector;
	private long time;
	
	public Cluster(Vector v, long time) {
		vectorList = new ArrayList<Vector>();
		vectorList.add(v);
		this.centerVector = v;
		this.time = time;
	}
	
	
	
	
	public long getTime() {
		return time;
	}
	
	public Vector getCenterVector() {
		return centerVector;
	}
	
	public void addVector(Vector v) {
		vectorList.add(v);
	}
	
	public void resetCenterVector() {
		int size = centerVector.size();
		double[] sumArray = new double[size];
		for(int i=0 ; i<size ; ++i) {
			sumArray[i] = 0;
		}
		
		for(Vector v : vectorList) {
			double[] a = v.toArray();
			for(int i=0 ; i<size ; ++i) {
				sumArray[i] += a[i];
			}
		}
		
		for(int i=0 ; i<size ; ++i) {
			sumArray[i] /= vectorList.size();
		}
		
		centerVector = new DenseVector(sumArray);
	}
}
