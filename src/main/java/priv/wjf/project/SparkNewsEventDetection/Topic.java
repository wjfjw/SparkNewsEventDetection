package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

public class Topic 
{
	private List<Event> eventList;
	private List<Integer> eventIdList;
	private Vector centerVector;
	
	public Topic(Event event, int eventId) {
		eventList = new ArrayList<Event>();
		eventList.add(event);
		
		eventIdList = new ArrayList<Integer>();
		eventIdList.add(eventId);
		this.centerVector = event.getCenterVector();
	}
	
	//eventList
	public List<Event> getEventList() {
		return eventList;
	}
	
	public void addEvent(Event event) {
		eventList.add(event);
	}
	
	//eventIdList
	public List<Integer> getEventIdList() {
		return eventIdList;
	}
	
	public void addEventId(int eventId) {
		eventIdList.add(eventId);
	}
	
	//centerVector
	public Vector getCenterVector() {
		return centerVector;
	}
	
	public void resetCenterVector() {
		int size = centerVector.size();
		double[] sumArray = new double[size];
		for(int i=0 ; i<size ; ++i) {
			sumArray[i] = 0;
		}
		
		for(Event event : eventList) {
			Vector v = event.getCenterVector();
			double[] a = v.toArray();
			for(int i=0 ; i<size ; ++i) {
				sumArray[i] += a[i];
			}
		}
		
		for(int i=0 ; i<size ; ++i) {
			sumArray[i] /= eventList.size();
		}
		
		centerVector = new DenseVector(sumArray).toSparse();
	}
}
