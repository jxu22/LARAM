package aspectSegmenter;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Map;
import java.util.Vector;

import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import opennlp.tools.util.Span;
import aspectSegmenter.Review.Sentence;
import aspectSegmenter.Review.Token;

public class Analyzer {
	//this aspect set only exist in the old TripAdvisor reviews
	//public static final String[] ASPECT_SET = {"Value", "Room", "Location", "Cleanliness", "Frontdesk", "Service", "Business Service"};
	//this is the common aspect set in TripAdvisor reviews
	public static final String[] ASPECT_SET_NEW = {"environment", "taste", "price"};
	public static final int ASPECT_COUNT_CUT = 0;
	public static final int ASPECT_CONTENT_CUT = 0;
	public static final String PUNCT = ":;=+-()[],.\"'";
	
	class _Aspect{
		String m_name;
		HashSet<String> m_keywords;
		
		_Aspect(String name, HashSet<String> keywords){
			m_name = name;
			m_keywords = keywords;
		}
	}
	
	Vector<Hotel> m_hotelList;	
	//TODO: The m_keywords is fixed, different from the m_keywords variable in _Aspect class..
	Vector<_Aspect> m_keywords;
	Hashtable<String, Integer> m_vocabulary;//indexed vocabulary
	Vector<String> m_wordlist;//term list in the original order
	
	HashSet<String> m_stopwords;
	Vector<rank_item<String>> m_ranklist;
	double[][] m_chi_table;
	double[] m_wordCount;
	boolean m_isLoadCV; // if the vocabulary is fixed
	
	//specific parameter to be tuned for bootstrapping aspect segmentation
	static public double chi_ratio = 4.0;
	static public int chi_size = 35;
	static public int chi_iter = 10;
	static public int tf_cut = 10;
	
	//NLP modules
	SentenceDetectorME m_stnDector;
	TokenizerME m_tokenizer;
	POSTaggerME m_postagger;
	Stemmer m_stemmer;	
		
	class rank_item<E> implements Comparable<rank_item<E>>{
		E m_name;
		double m_value;
		
		public rank_item(E name, double value){
			m_name = name;
			m_value = value;
		}
		
		@Override
		public int compareTo(rank_item<E> v) {
			if (m_value < v.m_value) return 1;
			else if (m_value > v.m_value) return -1;
			return 0;
		}
		
	}
	
	public Analyzer(String seedwords, String stopwords, String stnSplModel, String tknModel, String posModel){
		m_hotelList = new Vector<Hotel>();
		m_vocabulary = new Hashtable<String, Integer>();
		m_chi_table = null;
		m_isLoadCV = false;
		if (seedwords != null && seedwords.isEmpty()==false)
			LoadKeywords(seedwords);
		LoadStopwords(stopwords);
		
		try {
			m_stnDector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnSplModel)));
			m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tknModel)));
			m_postagger = new POSTaggerME(new POSModel(new FileInputStream(posModel)));
			m_stemmer = new Stemmer();
		} catch (InvalidFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("[Info]NLP modules initialized...");
	}
	
	/**
	 * The key word(seeds) loading is dynamic, which means #aspect is not fixed
	 * @param filename
	 */
	public void LoadKeywords(String filename){
		try {
			m_keywords = new Vector<_Aspect>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			String[] container;
			HashSet<String> keywords;
			while( (tmpTxt=reader.readLine()) != null ){
				container = tmpTxt.split(" ");
				keywords = new HashSet<String>(container.length-1);
				for(int i=1; i<container.length; i++)
					keywords.add(container[i]);
				m_keywords.add(new _Aspect(container[0], keywords));
				System.out.println("Keywords for " + container[0] + ": " + keywords.size());
			}
			reader.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void LoadVocabulary(String filename){
		try {
			m_vocabulary = new Hashtable<String, Integer>();
			m_wordlist = new Vector<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			String[] container;
			while( (tmpTxt=reader.readLine()) != null ){
				container = tmpTxt.split("\t");
				m_vocabulary.put(container[0], m_vocabulary.size());
				m_wordlist.add(tmpTxt.trim());
			}
			reader.close();
			m_isLoadCV = true;
			System.out.println("[Info]Load " + m_vocabulary.size() + " control terms...");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void LoadStopwords(String filename){
		try {
			m_stopwords = new HashSet<String>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			while( (tmpTxt=reader.readLine()) != null )
				m_stopwords.add(tmpTxt.toLowerCase());
			reader.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
		
	public String[] getLemma(String[] tokens){
		String[] lemma = new String[tokens.length];
		String term;
		for(int i=0; i<lemma.length; i++){
			//lemma[i] = m_stemmer.stem(tokens[i].toLowerCase());//shall we stem it?
			term = tokens[i].toLowerCase();
			if (term.length()>1 && PUNCT.indexOf(term.charAt(0))!=-1 && term.charAt(1)>='a' && term.charAt(1)<='z')
				lemma[i] = term.substring(1);
			else 
				lemma[i] = term;
		}
		return lemma;
	}
	
	static public String getHotelID(String fname){
		int start = fname.indexOf("hotel_"), end = fname.indexOf(".dat");
		if (start==-1)
			return fname.substring(0, end);
		else
			return fname.substring(start+"hotel_".length(), end);
	}
	
	private String cleanReview(String content){
		String error_A = "showReview\\([\\d]+\\, [\\w]+\\);";//
		return content.replace(error_A, "");
	}
	
	/**
	 * The review format is fixed
	 * The function does the following things:
	 * 1. parse the hotel data and load them into vector of hotel
	 * 2. use opennlp to tokenize the reviews(create sentences...)
	 * 3. expand the vocabulary
	 * @param filename
	 */
	public void LoadReviews(String filename){//load reviews for annotation purpose
		try {
			File f = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"));
			String tmpTxt, fname = getHotelID(f.getName()), title = "", content = null;
			int review_size = 0;
			
			Review review = null;
			String[] stns, tokens;
			Span[] stn_spans;
			int[] ratings = new int[1+ASPECT_SET_NEW.length];
			Hotel tHotel = new Hotel(fname);
			while((tmpTxt=reader.readLine()) != null){
				if (tmpTxt.startsWith("<Title>"))
		    		title = tmpTxt.substring("<Title>".length()+1, tmpTxt.length()-1);
				else if (tmpTxt.startsWith("<Overall>")){//only read those aspects
					try{
			    		double r = Double.valueOf(tmpTxt.substring("<Overall>".length()));
			    		ratings[0] = (int)r;
					} catch (Exception e){
						System.err.println("Error format: " + fname);
						reader.close();
						return;
					}
		    	}
		    	else if (tmpTxt.startsWith("<Value>"))
		    		ratings[1] = Integer.valueOf(tmpTxt.substring("<Value>".length()));
		    	else if (tmpTxt.startsWith("<Rooms>"))
		    		ratings[2] = Integer.valueOf(tmpTxt.substring("<Rooms>".length()));
		    	else if (tmpTxt.startsWith("<Location>"))
		    		ratings[3] = Integer.valueOf(tmpTxt.substring("<Location>".length()));
		    	else if (tmpTxt.startsWith("<Cleanliness>"))
		    		ratings[4] = Integer.valueOf(tmpTxt.substring("<Cleanliness>".length()));
		    	else if (tmpTxt.startsWith("<Service>"))
		    		ratings[5] = Integer.valueOf(tmpTxt.substring("<Service>".length()));
				else if (tmpTxt.startsWith("<Content>"))
					content = cleanReview(tmpTxt.substring("<Content>".length()));
				else if (tmpTxt.isEmpty() && content != null){
				    // TODO: this method detect the position of the first words of a set of sentences
					stn_spans = m_stnDector.sentPosDetect(content);//list of the sentence spans
					if (stn_spans.length<3){
						content = null;
						Arrays.fill(ratings, 0);
						continue;
					}
					
					stns = Span.spansToStrings(stn_spans, content);
					//TODO: I have no idea what is the use of the aspect rating from the tripadvisor
					review = new Review(fname, Integer.toString(review_size), ratings);
					for(int i=0; i<stns.length; i++){
						tokens = m_tokenizer.tokenize(stns[i]);
						if (tokens!=null && tokens.length>2)//discard too short sentences
							review.addStn(stns[i], tokens, m_postagger.tag(tokens), getLemma(tokens), m_stopwords);
				    }
					
					if (review.getStnSize()>2){
						if (title.isEmpty()==false){//include the title as content
							tokens = m_tokenizer.tokenize(title);
							if (tokens!=null && tokens.length>2)//discard too short sentences
								review.addStn(title, tokens, m_postagger.tag(tokens), getLemma(tokens), m_stopwords);
						}
						
						if (m_isLoadCV==false)//didn't load the controlled vocabulary
							expendVocabular(review);
						tHotel.addReview(review);
						review_size ++;
					}
					
					content = null;
					Arrays.fill(ratings, 0);
				}
			}
			reader.close();
			
			if (tHotel.getReviewSize()>1){
				m_hotelList.add(tHotel);
				if (m_hotelList.size()%100==0)
					System.out.print(".");
				if (m_hotelList.size()%10000==0)
					System.out.println(".");
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	
	public void LoadDirectory(String path, String suffix){		
	    //System.out.println("Start loading reviews from " + path + " ...");
		File dir = new File(path);
		int size = m_hotelList.size();
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix))
				LoadReviews(f.getAbsolutePath());
			else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
		size = m_hotelList.size() - size;
		System.out.println("Loading " + size + " hotels from " + path);
	}
	
	//save for hReviews
	public void Save2Vectors(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "US-ASCII"));
			int[][] vectors = new int[m_keywords.size()][m_vocabulary.size()];
			double[] ratings = new double[1+m_keywords.size()], counts = new double[1+m_keywords.size()];
			int aspectID, wordID, outputSize=0, reviewSize=0;
			for(Hotel hotel:m_hotelList){
				for(Review r:hotel.m_reviews){//aggregate all the reviews
					Annotate(r);
					
					reviewSize++;
					for(aspectID=0; aspectID<ratings.length; aspectID++){
						if (r.m_ratings[aspectID]>0){
							ratings[aspectID] += r.m_ratings[aspectID];
							counts[aspectID] += 1;//only take the average in the existing ratings
						}
					}
					
					//collect the vectors
					for(Sentence stn:r.m_stns){
						if ((aspectID = stn.m_aspectID)<0)
							continue;
						
						for(Token t:stn.m_tokens){//select the in-vocabulary word
							if (m_vocabulary.containsKey(t.m_lemma)){
								wordID = m_vocabulary.get(t.m_lemma);
								vectors[aspectID][wordID]++;
							}
						}
					}
				}					
				
				if (ready4output(vectors, counts)){
					Save2Vector(writer, hotel.m_ID, reviewSize, ratings, counts, vectors);
					outputSize ++;
				}
				clearVector(ratings, counts, vectors);
			}
			
			writer.close();
			System.out.println("Output " + outputSize + " hotel-reviews...");
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void Save2Vector(BufferedWriter writer, String hotelID, int reviewSize, double[] ratings, double[] counts, int[][] vectors) throws IOException{
		int aspectID, wordID;
		DecimalFormat formater = new DecimalFormat("#.###");
		writer.write(hotelID);
		double score;
		for(aspectID=0; aspectID<ratings.length; aspectID++){
			if (counts[aspectID]>0)
				score = ratings[aspectID]/counts[aspectID];
			else 
				score = ratings[0]/counts[0];//using overall rating as default
			writer.write("\t" + formater.format(score));
		}
		writer.write("\n");
		
		for(aspectID=0; aspectID<vectors.length; aspectID++){
			for(wordID=0; wordID<vectors[aspectID].length; wordID++){
				if (vectors[aspectID][wordID]>0)
					writer.write(wordID + ":" + vectors[aspectID][wordID] + " ");
			}
			writer.write("\n");
		}
	}
	
	private void clearVector(double[] ratings, double[] counts, int[][] vectors){
		Arrays.fill(ratings, 0);
		Arrays.fill(counts, 0);
		for(int aspectID=0; aspectID<vectors.length; aspectID++)
			Arrays.fill(vectors[aspectID], 0);
	}
	
	//more strategies can be derived for selecting the reviews
	private boolean ready4output(int[][] vectors, double[] counts){
		for(int aspectID=0; aspectID<vectors.length; aspectID++){
			if (counts[aspectID]<=ASPECT_COUNT_CUT)
				return false;//at least have these amount of user ratings
			
			double sum = 0;
			for(int wordID=0; wordID<vectors[aspectID].length; wordID++)
				sum += vectors[aspectID][wordID];
			if (sum<=ASPECT_CONTENT_CUT)
				return false;//at least have these amount of words in content
		}
		return true;
	}
	
	private void expendVocabular(Review tReview){
		for(Sentence stn : tReview.m_stns){
			for(Token t : stn.m_tokens){
				if (m_vocabulary.containsKey(t.m_lemma) == false)
					m_vocabulary.put(t.m_lemma, m_vocabulary.size());
			}
		}
	}
	
	private void createChiTable(){
		if (m_chi_table==null){
			m_chi_table = new double[m_keywords.size()][m_vocabulary.size()];
			m_wordCount = new double[m_vocabulary.size()];
			
			m_ranklist = new Vector<rank_item<String>>(m_vocabulary.size());
			Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();
			while(vIt.hasNext()){
				Map.Entry<String, Integer> entry = vIt.next();
				m_ranklist.add(new rank_item<String>(entry.getKey(), 0.0));
			}
		}
		else{
			for(int i=0; i<m_chi_table.length; i++)
				Arrays.fill(m_chi_table[i], 0.0);
			Arrays.fill(m_wordCount, 0.0);
		}
	}
	
	private void Annotate(Review tReview){
		for(Sentence stn : tReview.m_stns){
			int maxCount = 0, count, sel = -1;
			for(int index=0; index<m_keywords.size(); index++){				
				if ( (count=stn.AnnotateByKeyword(m_keywords.get(index).m_keywords))>maxCount ){
					maxCount = count;
					sel = index;
				}
				else if (count==maxCount)
					sel = -1;//don't allow tie
			}
			stn.setAspectID(sel);
		}
	}
	
	private void collectStats(Review tReview){
		int aspectID, wordID;
		for(Sentence stn : tReview.m_stns){
			if ( (aspectID=stn.getAspectID())>-1){
				for(Token t:stn.m_tokens){
					if (m_vocabulary.containsKey(t.m_lemma) == false)
						System.out.println("Missing:" + t);
					else{
						wordID = m_vocabulary.get(t.m_lemma);
						m_chi_table[aspectID][wordID] ++;
					}
				}
			}
		}
	}
	
	/**
	 * 
	 * @param A: w and c
	 * @param B: w and !c
	 * @param C: !w and c
	 * @param D: !w and !c
	 * @param N: total
	 * @return Chi-Sqaure
	 */
	private double ChiSquareValue(double A, double B, double C, double D, double N){
		double denomiator = (A+C) * (B+D) * (A+B) * (C+D);
		if (denomiator>0 && A+B > tf_cut)
			return N * (A*D-B*C) * (A*D-B*C) / denomiator;
		else
			return 0.0;//problematic setting (this word hasn't been assigned)
	}
	
	private void ChiSquareTest(){		
		createChiTable();
		for(Hotel hotel:m_hotelList){
			for(Review tReview:hotel.m_reviews){
				Annotate(tReview);
				collectStats(tReview);
			}
		}
		
		double N = 0;
		double[] aspectCount = new double[m_keywords.size()];
		int i, j;
		for(i=0; i<aspectCount.length; i++){
			for(j=0; j<m_wordCount.length; j++){
				aspectCount[i] += m_chi_table[i][j];
				m_wordCount[j] += m_chi_table[i][j];
				N += m_chi_table[i][j];
			}
		}
		
		for(i=0; i<aspectCount.length; i++){
			for(j=0; j<m_wordCount.length; j++){
				m_chi_table[i][j] = ChiSquareValue(m_chi_table[i][j], 
													m_wordCount[j]-m_chi_table[i][j], 
													aspectCount[i]-m_chi_table[i][j],
													N-m_chi_table[i][j], N);
			}
		}
	}
	
	private void getVocabularyStat(){
		if (m_wordCount==null)
			m_wordCount = new double[m_vocabulary.size()];
		else
			Arrays.fill(m_wordCount, 0);
		
		for(Hotel hotel:m_hotelList){
			for(Review tReview:hotel.m_reviews){
				for(Sentence stn : tReview.m_stns){					
					for(Token t:stn.m_tokens){
						if (m_vocabulary.containsKey(t.m_lemma) == false){
							System.out.println("Missing:" + t);
						} else{
							int wordID = m_vocabulary.get(t.m_lemma);
							m_wordCount[wordID] ++;
						}
					}
				}
			}
		}
	}
	
	private boolean expandKeywordsByChi(double ratio){
		int wordID, aspectID, selID = -1;
		double maxChi, chiV;
		Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();
		while(vIt.hasNext()){//first iteration, select the maxAspect
			Map.Entry<String, Integer> entry = vIt.next();
			wordID = entry.getValue();
			
			maxChi = 0.0;
			selID = -1;
			for(aspectID=0; aspectID<m_keywords.size(); aspectID++){
				if ((chiV=m_chi_table[aspectID][wordID]) > ratio * maxChi){
					maxChi = chiV;
					selID = aspectID;
				}
			}
			
			for(aspectID=0; aspectID<m_keywords.size(); aspectID++){
				if (aspectID!=selID)
					m_chi_table[aspectID][wordID] = 0.0;
			}
		}
		
		aspectID = 0;
		boolean extended = false;
		for(int i=0; i<m_keywords.size(); i++){	
			_Aspect asp = m_keywords.get(i);
			for(rank_item<String> item : m_ranklist){//second iteration, select the maxAspect
				wordID = m_vocabulary.get(item.m_name);
				item.m_value = m_chi_table[aspectID][wordID];	
			}
			
			Collections.sort(m_ranklist);
			for(wordID=0; wordID<chi_size; wordID++){
				if (asp.m_keywords.add(m_ranklist.get(wordID).m_name))
					extended = true;
			}
			aspectID ++;
		}
		return extended;
	}
	
	public void OutputChiTable(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			int wordID, aspectID;
			for(aspectID=0; aspectID<m_keywords.size(); aspectID++)
				writer.write("\t" + m_keywords.get(aspectID).m_name);
			writer.write("\n");
			
			Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();			
			while(vIt.hasNext()){
				Map.Entry<String, Integer> entry = vIt.next();
				wordID = entry.getValue();
				writer.write(entry.getKey());
				for(aspectID=0; aspectID<m_keywords.size(); aspectID++)
					writer.write("\t" + m_chi_table[aspectID][wordID]);
				writer.write("\n");
			}
			writer.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void SaveVocabulary(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			getVocabularyStat();
			
			Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();
			while(vIt.hasNext()){//iterate over all the words
				Map.Entry<String, Integer> entry = vIt.next();
				int wordID = entry.getValue();
				
				writer.write(entry.getKey() + "\t" + m_wordCount[wordID] + "\n");
			}
			writer.close();
			
			System.out.println("[Info]Vocabulary size: " + m_vocabulary.size());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//output word list with more statistic info: CHI, DF 
	public void OutputWordListWithInfo(String filename){
		System.out.println("Vocabulary size: " + m_vocabulary.size());
		try {
			ChiSquareTest();//calculate the chi table
			
			int wordID, aspectID;
			double chi_value; 
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			Iterator<Map.Entry<String, Integer>> vIt = m_vocabulary.entrySet().iterator();
			while(vIt.hasNext()){//iterate over all the words
				Map.Entry<String, Integer> entry = vIt.next();
				wordID = entry.getValue();
				
				chi_value = 0;
				for(aspectID=0; aspectID<m_keywords.size(); aspectID++)
					chi_value += m_chi_table[aspectID][wordID];//calculate the average Chi2
				
				if (chi_value/aspectID>3.84 && m_wordCount[wordID]>50)
					writer.write(entry.getKey() + "\t" + (chi_value/aspectID) + "\t" + m_wordCount[wordID] + "\n");
			}
			writer.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void BootStrapping(String filename){
		System.out.println("Vocabulary size: " + m_vocabulary.size());
		
		int iter = 0;
		do {
			ChiSquareTest();
			System.out.println("Bootstrapping for " + iter + " iterations...");
		}while(expandKeywordsByChi(chi_ratio) && ++iter<chi_iter );
		
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			for(int i=0; i<m_keywords.size(); i++){
				_Aspect asp = m_keywords.get(i);
				writer.write(asp.m_name);
				Iterator<String> wIter = asp.m_keywords.iterator();
				while(wIter.hasNext())
					writer.write(" " + wIter.next());
				writer.write("\n");
			}
			writer.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	static public void main(String[] args){
		Analyzer analyzer = new Analyzer("Data/Seeds/hotel_bootstrapping.dat", "Data/Seeds/stopwords.dat", 
				"Data/Model/NLP/en-sent.zip", "Data/Model/NLP/en-token.zip", "Data/Model/NLP/en-pos-maxent.bin");
		//analyzer.LoadVocabulary("Data/Seeds/hotel_vocabulary_CHI.dat");
		analyzer.LoadDirectory("Data/Reviews/", ".dat");
		//analyzer.LoadReviews("e:/Data/Reviews/Tmp/hotel_111849.dat");
		analyzer.BootStrapping("Data/Seeds/hotel_bootstrapping_test.dat");
		//analyzer.OutputWordListWithInfo("Data/Seeds/hotel_vocabulary_May10.dat");
		analyzer.Save2Vectors("Data/Vectors/vector_CHI_4000.dat");	
		//analyzer.SaveVocabulary("Data/Seeds/hotel_vocabulary.dat");
	}
}
