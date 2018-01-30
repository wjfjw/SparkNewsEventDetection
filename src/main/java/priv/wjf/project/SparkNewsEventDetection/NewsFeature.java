package priv.wjf.project.SparkNewsEventDetection;

import org.apache.spark.mllib.linalg.Vector;

public class NewsFeature 
{
	private String id;
	private Vector vector;
	
	public NewsFeature(String id, Vector vector) {
		this.id = id;
		this.vector = vector;
	}
	
	public String getId() {
		return id;
	}
	
	public Vector getVector() {
		return vector;
	}
}
