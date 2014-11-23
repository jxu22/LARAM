package utilities;

import java.util.Random;

public class Utilities {
	static Random rand = new Random();
	
	public static void randomize(double[] v){
		for(int i=0; i<v.length; i++)
			v[i] = (2*rand.nextDouble()-1)/10;
	}
	
	public static double expSum(double[] values){
		double sum = 0;
		for(double v:values)
			sum += Math.exp(v);
		return sum;
	}
	
	public static double sum(double[] values){
		double sum = 0;
		for(double v:values)
			sum += v;
		return sum;
	}
	
	public static double MSE(double[] pred, double[] answer, int offset){
		double mse = 0;
		for(int i=0; i<pred.length; i++)
			mse += (pred[i]-answer[i+offset]) * (pred[i]-answer[i+offset]);
		return mse/pred.length;
	}
	
	public static double correlation(double[] pred, double[] answer, int offset){
		double m_x = 0, m_y = 0, s_x = 0, s_y = 0;
		
		//first order moment
		for(int i=0; i<pred.length; i++){
			m_x += pred[i];
			m_y += answer[i+offset];
		}
		m_x /= pred.length;
		m_y /= pred.length;
		
		//second order moment
		for(int i=0; i<pred.length; i++){
			s_x += (pred[i]-m_x) * (pred[i]-m_x);
			s_y += (answer[offset+i]-m_y) * (answer[offset+i]-m_y);
		}
		
		//handle special cases
		if (s_x==0 && s_y==0)
			return 1;
		else if (s_x==0 || s_y==0)
			return 0;
		
		s_x = Math.sqrt(s_x/(pred.length-1));
		s_y = Math.sqrt(s_y/(pred.length-1));
		
		//Pearson correlation
		double correlation = 0;
		for(int i=0; i<pred.length; i++)
			correlation += (pred[i]-m_x)/s_x * (answer[offset+i]-m_y)/s_y;
		return correlation/(pred.length-1.0);
	}
}
