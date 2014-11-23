package lara;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Random;

import utilities.Utilities;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

public class LRR_Model {
	int m_k; // # of aspects
	int m_v; // # of words
	double[] m_mu; // prior for \alpha in each review
	double[][] m_sigma_inv; // precision matrix (NOT covariance!)
	DenseDoubleMatrix2D m_sigma; // only used for calculating inverse(\Sigma)
	double[][] m_beta; // word sentiment polarity matrix should have one bias term!
	double m_delta; // variance of overall rating prediction (\sigma in the manual)
	
	Algebra m_alg;
	
	public LRR_Model(int k, int v){
		m_k = k;
		m_v = v;

		init();
	}
	
	public LRR_Model(String filename){
		LoadFromFile(filename);
	}
	
	private void create(){
		m_mu = new double[m_k];
		m_sigma = new DenseDoubleMatrix2D(m_k, m_k);
		m_sigma_inv = new double[m_k][m_k];
		m_beta = new double[m_k][m_v+1];
		
		m_alg = new Algebra();
	}
	
	protected void init(){
		create();
		
		Random rand = new Random();
		for(int i=0; i<m_k; i++){
			m_mu[i] = (2.0*rand.nextDouble() - 1.0);
			m_sigma_inv[i][i] = 1.0;
			m_sigma.setQuick(i, i, 1.0);
			Utilities.randomize(m_beta[i]);
		}
		m_delta = 1.0;
	}
	
	public double calcCovariance(double[] vct){
		double sum = 0, s;
		for(int i=0; i<m_k; i++){
			s = 0;
			for(int j=0; j<m_k; j++)
				s += vct[j] * m_sigma_inv[j][i];
			sum += s * vct[i];
		}
		return sum;
	}
	
	public double calcDet(){
		return m_alg.det(m_sigma);
	}
	
	public void calcSigmaInv(double scale){
		DoubleMatrix2D inv = m_alg.inverse(m_sigma);
		for(int i=0; i<m_k; i++){
			for(int j=0; j<m_k; j++)
				m_sigma_inv[i][j] = inv.getQuick(i, j) * scale;
		}
	}
	
	public void Save2File(String filename){
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			writer.write(m_k + "\t" + m_v + "\n");
			
			//save \mu
			for(int i=0; i<m_k; i++)
				writer.write(m_mu[i] + "\t");
			writer.write("\n");
			
			//save \sigma
			for(int i=0; i<m_k; i++){
				for(int j=0; j<m_k; j++)
					writer.write(m_sigma.getQuick(i, j) + "\t");
				writer.write("\n");
			}
			
			//save \beta
			for(int i=0; i<m_k; i++){
				for(int j=0; j<=m_v; j++)
					writer.write(m_beta[i][j] + "\t");
				writer.write("\n");
			}
			
			//save delta
			writer.write(Double.toString(m_delta));
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	private void LoadFromFile(String filename){
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			String[] container;
			
			//part 1: aspect size, vocabulary size
			tmpTxt=reader.readLine();
			container = tmpTxt.split("\t");
			m_k = Integer.valueOf(container[0]);
			m_v = Integer.valueOf(container[1]);
			create();
			
			//part 2: \mu
			tmpTxt=reader.readLine();
			container = tmpTxt.split("\t");
			for(int i=0; i<m_k; i++)
				m_mu[i] = Double.valueOf(container[i]);
			
			//part 3: \sigma
			for(int i=0; i<m_k; i++){
				tmpTxt=reader.readLine();
				container = tmpTxt.split("\t");
				for(int j=0; j<m_k; j++){
					m_sigma.setQuick(i, j, Double.valueOf(container[j]));
				}	
			}
			calcSigmaInv(1.0);
			
			//part 4: \beta
			for(int i=0; i<m_k; i++){
				tmpTxt=reader.readLine();
				container = tmpTxt.split("\t");
				for(int j=0; j<=m_v; j++){
					m_beta[i][j] = Double.valueOf(container[j]);
				}
			}
			
			//part 5: \delta
			tmpTxt=reader.readLine();
			m_delta = Double.valueOf(tmpTxt.trim());
			
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
