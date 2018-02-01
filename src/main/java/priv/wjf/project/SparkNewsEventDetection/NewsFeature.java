package priv.wjf.project.SparkNewsEventDetection;

import org.apache.spark.mllib.linalg.Vector;

public class NewsFeature 
{
	private int id;
	private long time;
	private Vector vector;
	
	public NewsFeature(int id, long time, Vector vector) {
		this.id = id;
		this.time = time;
		this.vector = vector;
	}
	
	public int getId() {
		return id;
	}
	
	public long getTime() {
		return time;
	}
	
	public Vector getVector() {
		return vector;
	}
}
