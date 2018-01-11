package priv.wjf.project.SparkNewsEventDetection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.CoordinateMatrix;
import org.apache.spark.mllib.linalg.distributed.IndexedRow;
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix;
import org.apache.spark.mllib.linalg.distributed.MatrixEntry;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

public class FeatureExtraction 
{
	
	public static JavaRDD<Vector> getTfidfRDD(int dimension, JavaRDD<List<String>> segmentedLines)
	{
		//tf
		HashingTF hashingTF = new HashingTF(dimension);
		JavaRDD<Vector> tf =  hashingTF.transform(segmentedLines);
		tf.cache();
		
		//idf
		IDF idf = new IDF();
		IDFModel idfModel = idf.fit(tf);
		JavaRDD<Vector> tfidfRDD = idfModel.transform(tf);
		
		return tfidfRDD;
	}
	
	public static JavaRDD<Vector> getPCARDD(JavaRDD<Vector> tfidfRDD, int pcNum)
	{
		//PCA
		RowMatrix rowMatrix = new RowMatrix( tfidfRDD.rdd() );
		Matrix pc = rowMatrix.computePrincipalComponents(pcNum);
		RowMatrix dimreducedMatrix = rowMatrix.multiply(pc);
		JavaRDD<Vector> pcaRDD = dimreducedMatrix.rows().toJavaRDD();
		
		return pcaRDD;
	}
	
	public static void computeVectorSimilarity(JavaSparkContext sc, List<Vector> vectors) 
	{
		//计算特征向量之间的相似度
		List<IndexedRow> indexedRows = new ArrayList<IndexedRow>();
		for(int i=0 ; i<vectors.size() ; ++i) {
			indexedRows.add(new IndexedRow(i, vectors.get(i)));
		}
		IndexedRowMatrix indexedRowMatrix = new IndexedRowMatrix( sc.parallelize(indexedRows).rdd() );
		CoordinateMatrix similaryMatrix = indexedRowMatrix.toCoordinateMatrix().transpose().toRowMatrix().columnSimilarities();
		JavaRDD<MatrixEntry> matrixEntryRdd = similaryMatrix.entries().toJavaRDD();
		List<MatrixEntry> matrixEntries = matrixEntryRdd.collect();
				
		//输出特征向量之间的相似度
		System.out.println("\n*********************************");
		System.out.println("特征向量之间的相似度");
		for(MatrixEntry matrixEntry : matrixEntries) {
			System.out.println("(" + matrixEntry.i() + ", " + matrixEntry.j() + "):" + matrixEntry.value());
		}
		System.out.println("*********************************\n");		
	}

}
