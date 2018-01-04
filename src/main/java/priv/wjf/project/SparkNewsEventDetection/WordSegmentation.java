package priv.wjf.project.SparkNewsEventDetection;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.fnlp.nlp.cn.CNFactory;
import org.fnlp.nlp.cn.CNFactory.Models;
import org.fnlp.nlp.corpus.StopWords;
import org.fnlp.util.exception.LoadModelException;
import org.wltea.analyzer.core.IKSegmenter;
import org.wltea.analyzer.core.Lexeme;
import org.wltea.analyzer.lucene.IKAnalyzer;

public class WordSegmentation 
{
	private static String modelPath = "/home/wjf/JavaProject/SparkNewsEventDetection/lib/models/pku_seg";
	private static String stopWordsPath = "/home/wjf/JavaProject/SparkNewsEventDetection/lib/models/stopwords/StopWords.txt";
	private static CNFactory factory;
	
	static {
		try {
			// 创建中文处理工厂对象，并使用“models”目录下的模型文件初始化
			factory = CNFactory.getInstance(modelPath, Models.SEG);
		} catch (LoadModelException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * 使用FNLP分词
	 * @param content
	 * @return 分词后的词列表
	 */
	public static List<String> FNLPSegment(String content)
	{
		// 使用分词器对中文句子进行分词，得到分词结果
		String[] words = factory.seg(content);
		
		//去停用词
		StopWords sw = new StopWords(stopWordsPath);
     	List<String> wordList = sw.phraseDel(words);

		return wordList;
	}
	
	/**
	 * 使用IK独立分词，过滤停用词
	 * @param String content
	 * @return 分词后的词列表
	 */
	public static List<String> IKAloneSegment(String content)
	{
		Reader reader = new StringReader(content);
		List<String> wordList = new ArrayList<String>();
		IKSegmenter ikSegmenter = new IKSegmenter(reader, true);
		Lexeme lex = null;
		try {
			while( (lex = ikSegmenter.next()) != null ){
				String word = lex.getLexemeText();
				wordList.add(word);
//				if(!word.matches(".*\\d.*")){
//					wordList.add(word);
//    		    }
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return wordList;
	}
	
	
	/**
	 * 使用IK-Lucene分词，过滤停用词
	 * @param String content
	 * @return 分词后的词列表
	 */
	public static List<String> IKLuceneSegment(String content)
	{
		List<String> wordList = new ArrayList<String>();
		try 
		{
		    Reader reader = new StringReader(content);
		    Analyzer analyzer = new IKAnalyzer(true);
		    
		    TokenStream ts = analyzer.tokenStream( "" ,  reader );
		    CharTermAttribute ch = ts.addAttribute(CharTermAttribute.class);
		    ts.reset();
            while (ts.incrementToken()) {  
            	wordList.add(ch.toString());
            }  
            ts.end();  
            ts.close();
            analyzer.close();
        } 
		catch (Exception e) {
		    e.printStackTrace();
		   }
		
		return wordList;
	}
}
