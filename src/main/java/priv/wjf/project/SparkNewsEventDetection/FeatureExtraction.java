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
	
	private static SparkConf conf;
	private static JavaSparkContext sc;
//	private static SQLContext sqlContext;
	
	static
	{
		conf = new SparkConf()
				.setAppName("SparkNewsEventDetection")
				.setMaster("local")
				;
		
		sc = new JavaSparkContext(conf);
//		sqlContext = new SQLContext(sc);
	}

	public static void main(String[] args) 
	{
		JavaRDD<String> lines = sc.parallelize( Arrays.asList(
				"在实践中形成了以新发展理念为主要内容的习近平新时代中国特色社会主义经济思想",
				"在习近平新时代中国特色社会主义经济思想引领下，中国经济发展进入了新时代，由高速增长阶段转向高质量发展阶段",
				"2018考研数学出现 “神押题”，考生怀疑发生泄题。"
				) );
		JavaRDD<List<String>> segmentedLines = lines.map( (String line)-> {
			return WordSegmentation.FNLPSegment(line);
		});
		
		//输出分词结果
		System.out.println("\n++++++++++++++++++++++++++++++++");
		for(List<String> list : segmentedLines.collect()) {
			System.out.println(list);
		}
		System.out.println("++++++++++++++++++++++++++++++++\n");
		
		//tf
		HashingTF hashingTF = new HashingTF(50);
		JavaRDD<Vector> tf =  hashingTF.transform(segmentedLines);
		tf.cache();
		
		//idf
		IDF idf = new IDF();
		IDFModel idfModel = idf.fit(tf);
		JavaRDD<Vector> tfidf =  idfModel.transform(tf);
		
		//输出特征向量
		List<Vector> tfidfVectors = tfidf.collect();
		System.out.println("\n*********************************");
		System.out.println("特征向量");
		for(Vector v : tfidfVectors) {
			System.out.println(v);
		}
		System.out.println("*********************************\n");
		
		//计算特征向量之间的相似度
		computeVectorSimilarity(tfidfVectors);	
		
		
		//PCA
		RowMatrix rowMatrix = new RowMatrix(tfidf.rdd());
		Matrix pc = rowMatrix.computePrincipalComponents(5);
		RowMatrix dimreducedMatrix = rowMatrix.multiply(pc);
		List<Vector> pcaVectors = dimreducedMatrix.rows().toJavaRDD().collect();
		
		//输出PCA降维后的特征向量
		System.out.println("\n*********************************");
		System.out.println("PCA降维后的特征向量");
		for(Vector v : pcaVectors) {
			System.out.println(v);
		}
		System.out.println("*********************************\n");
		
		//计算特征向量之间的相似度
		computeVectorSimilarity(pcaVectors);
		
	}
	
	private static void computeVectorSimilarity(List<Vector> vectors) 
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
