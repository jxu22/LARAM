package lara;

import java.util.Random;

import utilities.Utilities;

public class Vector4Review {
	String m_ID;
	boolean m_4train;
	SpaVector[] m_aspectV;//the lengths are not necessary the same
	double[] m_ratings; // first position for overall rating
	
	double[] m_pred;// predicted aspect rating
	double[] m_pred_cache;// in case we need to map the calculated aspect rating
	double[] m_alpha;// inferred aspect weight
	double[] m_alpha_hat; // to map alpha into a simplex by logistic functions
	
	public Vector4Review(String ID, String[] ratings, boolean isTrain){
		m_ID = ID;
		m_4train = isTrain;
		
		m_ratings = new double[ratings.length];
		for(int i=0; i<ratings.length; i++)
			m_ratings[i] = Double.valueOf(ratings[i]);
		m_aspectV = new SpaVector[m_ratings.length-1];
		
		// structures for prediction
		m_pred = new double[m_aspectV.length];
		m_pred_cache = new double[m_aspectV.length];
		m_alpha = new double[m_aspectV.length];
		m_alpha_hat = new double[m_aspectV.length];
	}
	
	public int getAspectSize(){
		if (m_aspectV.length==1)
			return LRR.K;//overall rating only
		else
			return m_aspectV.length;
	}
	
	public void setAspect(int i, String[] features){
		m_aspectV[i] = new SpaVector(features);
	}
	
	public double getDocLength(){
		double sum = 0;
		for(SpaVector vct:m_aspectV)
			sum += vct.L1Norm();
		return sum;
	}
	
	public void normalize(){
		double norm = getDocLength(), aSize;
		Random rand = new Random();
		for(int i=0; i<m_aspectV.length; i++)
		{
			SpaVector vct = m_aspectV[i];
			
			aSize = vct.L1Norm();
			vct.normalize(aSize);
			m_alpha_hat[i] = rand.nextDouble() + Math.log(aSize / norm); // an estimate of aspect weight
		}
		
		norm = Utilities.expSum(m_alpha_hat);
		for(int i=0; i<m_aspectV.length; i++)
			m_alpha[i] = Math.exp(m_alpha_hat[i])/norm;
	}
	
	//get the largest word index in this document 
	public int getLength(){
		int len = 0;
		for(SpaVector vct:m_aspectV)
			len = Math.max(len, vct.getLength());
		return len;
	}
	
	public double getAspectSize(int k){
		return m_aspectV[k].L1Norm();
	}
	
	//apply model onto each aspect
	public void getAspectRating(double[][] beta){
		for(int i=0; i<m_aspectV.length; i++){
			m_pred_cache[i] = m_aspectV[i].dotProduct(beta[i]);
			if (RatingRegression.SCORE_SQUARE)// to avoid negative rating
				m_pred[i] = 0.5 * m_pred_cache[i] * m_pred_cache[i];
			else
				m_pred[i] = Math.exp(m_pred_cache[i]);
		}
	}
	
	public void getAspectRating(double[] beta, int v){
		for(int i=0; i<m_aspectV.length; i++){
			m_pred_cache[i] = m_aspectV[i].dotProduct(beta, v*i);
			if (RatingRegression.SCORE_SQUARE)// to avoid negative rating
				m_pred[i] = 0.5 * m_pred_cache[i] * m_pred_cache[i];
			else
				m_pred[i] = Math.exp(m_pred_cache[i]);
		}
	}
	
	public double dotProduct(double[] beta, int k){
		return m_aspectV[k].dotProduct(beta);//beta is the weight for k-th aspect
	}
}
