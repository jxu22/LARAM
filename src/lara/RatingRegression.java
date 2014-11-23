package lara;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import optimizer.LBFGS;
import optimizer.LBFGS.ExceptionWithIflag;
import utilities.Utilities;

public class RatingRegression {
	static final boolean SCORE_SQUARE = false;//rating will be map by s^2 or exp(s)
	static final boolean BY_OVERALL = false;//train aspect rating predictor by overall rating
	
	protected ArrayList<Vector4Review> m_collection;
	
	protected double[] m_diag_beta;// cached diagonal for beta inference
	protected double[] m_g_beta;// cached gradient for beta inference
	protected double[] m_beta;// long vector for the matrix of beta
	
	protected double[] m_diag_alpha;// cached diagonal for alpha inference
	protected double[] m_g_alpha;// cached gradient for alpha inference
	private double[] m_alpha; // cached for difference vector
	protected double[] m_alpha_cache; // to map alpha into a simplex by logistic functions
	
	protected int m_alphaStep;
	protected double m_alphaTol;
	protected int m_betaStep;
	protected double m_betaTol;
	protected double m_lambda;
	protected int m_v, m_k;
	protected int m_trainSize, m_testSize;
	
	protected Random m_rand;
	
	public RatingRegression(int alphaStep, double alphaTol, int betaStep, double betaTol, double lambda){
		m_alphaStep = alphaStep;
		m_alphaTol = alphaTol;
		
		m_betaTol = betaTol;
		m_betaStep = betaStep;
		m_lambda = lambda;
		
		m_collection = null;
		m_rand = new Random(0);//with fixed random seed in order to get the same train/test split
	}
	
	public int LoadVectors(String filename){
		return LoadVectors(filename, -1);
	}
	
	public int LoadVectors(String filename, int size){
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			
			m_trainSize = 0;
			m_testSize = 0;
			m_collection = new ArrayList<Vector4Review>();
			Vector4Review vct;
			int pos, len = 0;
			boolean isTrain;
			double[] aspectSize = null;
			while((tmpTxt=reader.readLine())!=null){
				pos = tmpTxt.indexOf('\t');
				if (isTrain = (m_rand.nextDouble()<0.75))//train/test splitting ratio 
					m_trainSize++;
				else
					m_testSize++;
				vct = new Vector4Review(tmpTxt.substring(0,pos), tmpTxt.substring(pos+1).split("\t"), isTrain);
				
				if (aspectSize==null){
					aspectSize = new double[vct.getAspectSize()];
					m_k = aspectSize.length;
				}
				
				
				for(int i=0; i<vct.getAspectSize(); i++){
					tmpTxt = reader.readLine();
					vct.setAspect(i, tmpTxt.split(" "));//different data format
					//vct.setAspect(i, tmpTxt.split("\t"));//depend on the input
					aspectSize[i] += vct.getAspectSize(i);
				}
				vct.normalize();
				
				m_collection.add(vct);
				len = Math.max(vct.getLength(), len);//max index word
				
				if (size>0 && m_collection.size()>=size)
					break;
			}
			
			double sum = Utilities.sum(aspectSize);
			System.out.print("[Info]Aspect length proportion:");
			for(double v:aspectSize)
				System.out.print(String.format("\t%.3f", v/sum));
			System.out.println();
			
			System.out.println("[Info]Load " + m_trainSize + "/" + m_testSize + " instances from " + filename + " with feature size " + len);
			reader.close();
			return len;
		} catch (IOException e) {
			e.printStackTrace();
			return 0;
		}
	}
	
	protected void evaluateAspect(){
		double aMSE = 0, oMSE = 0, icorr = 0, acorr = 0, corr, diff;
		int i = -1;
		boolean iError = false, aError = false;
		
		double[][] pred = new double[m_k][m_testSize], ans = new double[m_k][m_testSize];
		for(Vector4Review vct:m_collection){
			if (vct.m_4train)
				continue;//only evaluating in testing cases
			i ++;
			
			diff = prediction(vct) - vct.m_ratings[0];
			oMSE += diff*diff;
			for(int j=0; j<m_k; j++){
				pred[j][i] = vct.m_pred[j];
				ans[j][i] = vct.m_ratings[j+1];
			}
			
			//1. Aspect evaluation: to skip overall rating in ground-truth
			aMSE += Utilities.MSE(vct.m_pred, vct.m_ratings, 1);
			corr = Utilities.correlation(vct.m_pred, vct.m_ratings, 1);
			
			if (Double.isNaN(corr)==false)
				icorr += corr;
			else
				iError = true;//error occur
		}
		
		//2. entity level evaluation
		for(int j=0; j<m_k; j++){
			corr = Utilities.correlation(pred[j], ans[j], 0);
			if (Double.isNaN(corr)==false)
				acorr += corr;
			else
				aError = true;
		}
		
		//MSE for overall rating, MSE for aspect rating, item level correlation, aspect level correlation
		if (iError)
			System.out.print('x');
		else
			System.out.print('o');
		if (aError)
			System.out.print('x');
		else
			System.out.print('o');
		System.out.print(String.format(" %.3f\t%.3f\t%.3f\t%.3f", Math.sqrt(oMSE/m_testSize), Math.sqrt(aMSE/m_testSize), (icorr/m_testSize), (acorr/m_k)));
		
	}
	
	protected double init(int v){
		if (m_collection==null || m_collection.isEmpty()){
			System.err.println("[Error]Load training data first!");
			return -1;
		}
		
		Vector4Review vct = m_collection.get(0);
		m_v = v;
		m_k = vct.m_aspectV.length;
		
		m_diag_beta = new double[m_k * (m_v+1)];//to include the bias term for each aspect
		m_g_beta = new double[m_diag_beta.length];
		m_beta = new double[m_g_beta.length];
		
		m_diag_alpha = new double[m_k];
		m_g_alpha = new double[m_k];
		m_alpha = new double[m_k];
		m_alpha_cache = new double[m_k];
		
		return 0;
	}
	
	//using known aspect rating to learn the global aspect weight
	private double getRatingObjGradient(){
		double f = 0, orating, sum = Utilities.expSum(m_alpha);
		int i, j;
		
		for(i=0; i<m_k; i++)
		{
			m_alpha_cache[i] = Math.exp(m_alpha[i])/sum;
			m_g_alpha[i] = m_lambda * m_alpha[i];
			f += m_lambda * m_alpha[i] * m_alpha[i]; // diagonal co-variance
		}
				
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue;
			
			orating = -vct.m_ratings[0];
			for(i=0; i<m_k; i++)
				orating += m_alpha_cache[i] * vct.m_ratings[i+1];//estimate the overall rating
			
			f += orating * orating;//the difference
			for(i=0; i<m_k; i++){
				for(j=0; j<m_k; j++){
					if (j==i)
						m_g_alpha[i] += orating*vct.m_ratings[i+1] * m_alpha_cache[i]*(1-m_alpha_cache[i]);
					else
						m_g_alpha[i] -= orating*vct.m_ratings[j+1] * m_alpha_cache[i]*m_alpha_cache[j];
				}
			}
		}
		return f/2;
	}
	
	//estimate beta with known aspect ratings (specify aspect by k)
	private double getAspectObjGradient(int k, double[] beta){
		double f = 0, s, diff, rd;
		int j;
		SpaVector sVct;
		
		for(j=0; j<=m_v; j++)
		{			
			m_g_beta[j] = m_lambda * beta[j];
			f += m_lambda * beta[j] * beta[j];
		}
				
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue;
			
			s = vct.dotProduct(beta, k);
			rd = BY_OVERALL ? vct.m_ratings[0] : vct.m_ratings[k+1];//to get the ground-truth aspect ratings
			if (RatingRegression.SCORE_SQUARE){
				diff = 0.5*s*s - rd;
			} else {
				s = Math.exp(s);
				diff = s - rd;
			}
			f += diff * diff;//the difference
			diff *= s;
			
			sVct = vct.m_aspectV[k];
			m_g_beta[0] += diff;//for bias term
			for(j=0; j<sVct.m_index.length; j++)
				m_g_beta[sVct.m_index[j]] += diff*sVct.m_value[j];
		}
		return f/2;
	}
	
	protected double prediction(Vector4Review vct){
		//predict aspect rating
		vct.getAspectRating(m_beta, m_v+1);
		double orating = 0;
		for(int i=0; i<m_k; i++)
			orating += m_alpha_cache[i] * vct.m_pred[i];
		return orating;
	}
	
	public void EstimateAspectModel(String filename){
		init(LoadVectors(filename));
		
		double f = 0;
		int iflag[] = {0}, iprint [] = {-1,0}, n = 1+m_v, m = 5, icall = 0;
		double[] beta = new double[n];
		
		//training phase
		try {
			//Step 1: estimate rating regression model for overall rating with ground-truth aspect rating
			Arrays.fill(m_alpha, 0);
			Arrays.fill(m_diag_alpha, 0);
			do {
				f = getRatingObjGradient();//to be minimized
				LBFGS.lbfgs ( m_k , m , m_alpha , f , m_g_alpha , false , m_diag_alpha , iprint , m_alphaTol , 1e-20 , iflag );
			} while ( iflag[0] != 0 && ++icall <= m_alphaStep );			
			System.out.print("[Info]Model for overall rating converge to " + f + ", with the learnt weights:");
			f = Utilities.expSum(m_alpha);
			for(int i=0; i<m_k; i++){
				m_alpha_cache[i] = Math.exp(m_alpha[i])/f;//map to simplex for future use
				System.out.print(String.format("\t%.3f", m_alpha_cache[i]));
			}
			System.out.println();
			
			//Step 2: estimate rating regression model for each aspect with ground-truth aspect rating
			for(int i=0; i<m_k; i++){
				icall = 0;
				iflag[0] = 0;
				
				Utilities.randomize(beta);
				Arrays.fill(m_diag_beta, 0);
				do {
					f = getAspectObjGradient(i, beta);//to be minimized
					LBFGS.lbfgs ( n , m , beta , f , m_g_beta , false , m_diag_beta , iprint , m_betaTol , 1e-20 , iflag );
				} while ( iflag[0] != 0 && ++icall <= m_betaStep );
				System.out.println("[Info]Model for aspect_" + i + " converge to " + f);
				System.arraycopy(beta, 0, m_beta, i*n, n);
			}
			
			//testing phase
			System.out.println("[Info]oMSE\taMSE\taCorr\tiCorr");
			evaluateAspect();
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}
	}
	
	//save all the prediction results
	public void SavePrediction(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			for(Vector4Review vct:m_collection){
				writer.write(vct.m_ID);
				
				//all the ground-truth ratings
				for(int i=0; i<vct.m_ratings.length; i++)
					writer.write(String.format("\t%.3f", vct.m_ratings[i]));
				
				//predicted ratings
				vct.getAspectRating(m_beta, (1+m_v));
				writer.write("\t");
				for(int i=0; i<vct.m_pred.length; i++)
					writer.write(String.format("\t%.3f", vct.m_pred[i]));
				
				//inferred weights (not meaningful for baseline logistic regression)
				writer.write("\t");
				for(int i=0; i<vct.m_alpha.length; i++)
					writer.write(String.format("\t%.3f", vct.m_alpha[i]));
				writer.write("\n");
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void SaveModel(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			writer.write(m_k + "\t" + m_v + "\n");
			
			//\mu for \hat\alpha
			for(int i=0; i<m_k; i++)
				writer.write(m_alpha[i] + "\t");
			writer.write("\n");
			
			//\Sigma for \hat\alpha (unknown for logistic regression)
			for(int i=0; i<m_k; i++){
				for(int j=0; j<m_k; j++){
					if (i==j)
						writer.write("1.0\t");
					else
						writer.write("0.0\t");
				}
				writer.write("\n");
			}
			
			//\beta
			for(int i=0; i<m_k; i++){
				for(int j=0; j<=m_v; j++)
					writer.write(m_beta[i*(m_v+1) + j] + "\t");
				writer.write("\n");
			}
			
			//\sigma (unknown for logistic regression)
			writer.write("1.0");
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		RatingRegression model = new RatingRegression(500, 5e-2, 5000, 1e-4, 1.0);
		model.EstimateAspectModel("Data/Vectors/Vector_CHI_4000.dat");
		model.SavePrediction("Data/Results/prediction_baseline.dat");
		model.SaveModel("Data/Model/model_base_hotel.dat");
	}
}
