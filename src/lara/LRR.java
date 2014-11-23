package lara;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.Arrays;

import optimizer.LBFGS;
import optimizer.LBFGS.ExceptionWithIflag;
import utilities.Utilities;

public class LRR extends RatingRegression {
	static public boolean SIGMA = false;
	static public double PI = 0.5;
	static final public int K = 7;//if we need to manually set the aspect size

	public LRR_Model m_model; 
	protected double[] m_old_alpha; // in case optimization for alpha failed
	BufferedWriter m_trace;
	
	// aspect will be determined by the input file for LRR
	public LRR(int alphaStep, double alphaTol, int betaStep, double betaTol, double lambda){
		super(alphaStep, alphaTol, betaStep, betaTol, lambda);
		
		m_model = null;
		m_old_alpha = null;
	}
	
	// if we want to load previous models
	public LRR(int alphaStep, double alphaTol, int betaStep, double betaTol, double lambda, String modelfile){
		super(alphaStep, alphaTol, betaStep, betaTol, lambda);
		
		m_model = new LRR_Model(modelfile);
		m_old_alpha = new double[m_model.m_k];
	}
	
	@Override
	protected double init(int v){
		super.init(v);
		double initV = 1;// likelihood for the first iteration won't matter
		
		//keep track of the model update trace 
		try {
			m_trace = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("trace.dat"), "UTF-8"));
			for(int i=0; i<m_k; i++)
				m_trace.write(String.format("Aspect_%d\t" , i));
			m_trace.write("alpha\tbeta\tdata\taux_data\tsigma\n");//column title for the trace file
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		if (m_model==null){
			m_model = new LRR_Model(m_k, v);
			m_old_alpha = new double[m_model.m_k];
	
			PI = 2.0;//try to seek a better initialization of beta
			initV = MStep(false);//this is just estimated alpha, no need to update Sigma yet
			PI = 0.5;
		}
		
		return initV;
	}
	
	@Override
	protected double prediction(Vector4Review vct){
		//Step 1: infer the aspect ratings/weights
		EStep(vct);
		
		//Step 2: calculate the overall rating
		double orating = 0;
		for(int i=0; i<m_model.m_k; i++)
			orating += vct.m_alpha[i] * vct.m_pred[i];
		return orating;
	}
	
	protected double EStep(Vector4Review vct){
		//step 1: estimate aspect rating
		vct.getAspectRating(m_model.m_beta);
		
		//step 2: infer aspect weight
		try {
			System.arraycopy(vct.m_alpha, 0, m_old_alpha, 0, m_old_alpha.length);
			return infer_alpha(vct);
		} catch (ExceptionWithIflag e) {
			System.arraycopy(m_old_alpha, 0, vct.m_alpha, 0, m_old_alpha.length);//failed with exceptions
			return -2;
		}
	}
	
	//we are estimating \hat{alpha}
	protected double getAlphaObjGradient(Vector4Review vct){
		double expsum = Utilities.expSum(vct.m_alpha_hat), orating = -vct.m_ratings[0], s, sum = 0;
		
		// initialize the gradient
		Arrays.fill(m_g_alpha, 0);
		
		for(int i=0; i<m_model.m_k; i++){
			vct.m_alpha[i] = Math.exp(vct.m_alpha_hat[i])/expsum;//map to aspect weight
			
			orating += vct.m_alpha[i] * vct.m_pred[i];//estimate the overall rating
			m_alpha_cache[i] = vct.m_alpha_hat[i] - m_model.m_mu[i];//difference with prior
			
			s = PI*(vct.m_pred[i]-vct.m_ratings[0]) * (vct.m_pred[i]-vct.m_ratings[0]);
			
			if (Math.abs(s)>1e-10){//in case we will disable it
				for(int j=0; j<m_model.m_k; j++){
					if (j==i)
						m_g_alpha[j] += 0.5 * s * vct.m_alpha[i]*(1-vct.m_alpha[i]); 
					else
						m_g_alpha[j] -= 0.5 * s * vct.m_alpha[i]*vct.m_alpha[j];
				}
				sum += vct.m_alpha[i] * s;		
			}
		}
		
		double diff = orating/m_model.m_delta;
		for(int i=0; i<m_model.m_k; i++){
			s = 0;
			for(int j=0; j<m_model.m_k; j++){
				// part I of objective function: data likelihood
				if (i==j)
					m_g_alpha[j] += diff*vct.m_pred[i] * vct.m_alpha[i]*(1-vct.m_alpha[i]);
				else
					m_g_alpha[j] -= diff*vct.m_pred[i] * vct.m_alpha[i]*vct.m_alpha[j];
				
				// part II of objective function: prior
				s += m_alpha_cache[j] * m_model.m_sigma_inv[i][j];
			}
			
			m_g_alpha[i] += s;
			sum += m_alpha_cache[i] * s;
		}		
		
		return 0.5 * (orating*orating/m_model.m_delta + sum);
	}
	
	protected double infer_alpha(Vector4Review vct) throws ExceptionWithIflag{
		double f = 0;
		int iprint [] = {-1,0}, iflag[] = {0}, icall = 0, n = m_model.m_k, m = 5;

		//initialize the diagonal matrix
		Arrays.fill(m_diag_alpha, 0);
		do {
			f = getAlphaObjGradient(vct);//to be minimized
			LBFGS.lbfgs ( n , m , vct.m_alpha_hat , f , m_g_alpha , false , m_diag_alpha , iprint , m_alphaTol , 1e-20 , iflag );
		} while ( iflag[0] != 0 && ++icall <= m_alphaStep );
		
		if (iflag[0]!=0)
			return -1; // have not converged yet
		else{
			double expsum = Utilities.expSum(vct.m_alpha_hat);
			for(n=0; n<m_model.m_k; n++)
				vct.m_alpha[n] = Math.exp(vct.m_alpha_hat[n])/expsum;
			return f;
		}
	}
	
	private void testAlphaVariance(boolean updateSigma){
		try {
			int i;
			double v;
			
			//test the variance of \hat\alpha estimation
			Arrays.fill(m_diag_alpha, 0.0);
			for(Vector4Review vct:m_collection){
				if (vct.m_4train==false)
					continue; // do not touch testing cases
				
				for(i=0; i<m_k; i++){
					v = vct.m_alpha_hat[i] - m_model.m_mu[i];
					m_diag_alpha[i] += v * v; // just for variance
				}
			}
			
			for(i=0; i<m_k; i++){
				m_diag_alpha[i] /= m_trainSize;
				if (i==0 && updateSigma)
					m_trace.write("*");
				m_trace.write(String.format("%.3f:%.3f\t", m_model.m_mu[i], m_diag_alpha[i]));//mean and variance of \hat\alpha
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//m-step can only be applied to training samples!!
	public double MStep(boolean updateSigma){
		updateSigma = false; // shall we update Sigma?
		int i, j, k = m_model.m_k;
		double v;
		
		//Step 0: initialize the statistics
		Arrays.fill(m_g_alpha, 0.0);
		
		//Step 1: ML for \mu
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			for(i=0; i<k; i++)
				m_g_alpha[i] += vct.m_alpha_hat[i];
		}
		for(i=0; i<k; i++)
			m_model.m_mu[i] = m_g_alpha[i]/m_trainSize;
		testAlphaVariance(updateSigma);
		
		
		//Step 2: ML for \sigma
		if (updateSigma){//we may choose to not update \Sigma
			//clear up the cache
			for(i=0; i<k; i++)
				Arrays.fill(m_model.m_sigma_inv[i], 0);
			
			for(Vector4Review vct:m_collection){
				if (vct.m_4train==false)
					continue; // do not touch the testing cases
				
				for(i=0; i<k; i++)
					m_diag_alpha[i] = vct.m_alpha_hat[i] - m_model.m_mu[i];
				
				if(SIGMA){//estimate the whole covariance matrix
					for(i=0; i<k; i++){
						for(j=0; j<k; j++){
							m_model.m_sigma_inv[i][j] += m_diag_alpha[i] * m_diag_alpha[j];
						}
					}
				} else {// just for estimate diagonal
					for(i=0; i<k; i++)
						m_model.m_sigma_inv[i][i] += m_diag_alpha[i] * m_diag_alpha[i]; 
				}
			}
			
			for(i=0; i<k; i++){
				if (SIGMA){
					m_model.m_sigma_inv[i][i] = (1.0 + m_model.m_sigma_inv[i][i]) / (1 + m_trainSize); // prior
					for(j=0; j<k; j++)
						m_model.m_sigma.setQuick(i, j, m_model.m_sigma_inv[i][j]);
				} else {
					v = (1.0 + m_model.m_sigma_inv[i][i]) / (1 + m_trainSize);
					m_model.m_sigma.setQuick(i, i, v);
					m_model.m_sigma_inv[i][i] = 1.0 / v;
				}
			}
			m_model.calcSigmaInv(1);
		}
		
		//calculate the likelihood for the alpha part
		double alpha_likelihood = 0, beta_likelihood = 0;
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			for(i=0; i<k; i++)
				m_diag_alpha[i] = vct.m_alpha_hat[i] - m_model.m_mu[i];
			alpha_likelihood += m_model.calcCovariance(m_diag_alpha);
		}
		alpha_likelihood += Math.log(m_model.calcDet()); 
		
		//Step 3: ML for \beta
		try {
			ml_beta();
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}
		
		beta_likelihood = getBetaPriorObj();
		
		//Step 4: ML for \delta
		double datalikelihood = getDataLikelihood(), auxdata = getAuxDataLikelihood(), oldDelta = m_model.m_delta;
		m_model.m_delta = datalikelihood / m_trainSize;	
		datalikelihood /= oldDelta;
		
		try {
			m_trace.write(String.format("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", alpha_likelihood, beta_likelihood, datalikelihood, auxdata, Math.log(m_model.m_delta)));
		} catch (IOException e) {
			e.printStackTrace();
		}
		return alpha_likelihood + beta_likelihood + datalikelihood + auxdata + Math.log(m_model.m_delta);
	}
	
	//\beat^T * \beta
	protected double getBetaPriorObj(){
		double likelihood = 0;
		for(int i=0; i<m_model.m_beta.length; i++){
			for(int j=0; j<m_model.m_beta[i].length; j++)
				likelihood += m_model.m_beta[i][j] * m_model.m_beta[i][j];
		}
		return m_lambda * likelihood;
	}
	
	//\sum_d(\sum_i\alpha_{di}\S_{di}-r_d)^2/\sigma^2
	protected double getDataLikelihood(){
		double likelihood = 0, orating;
						
		// part I of objective function: data likelihood
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			orating = -vct.m_ratings[0];
			
			//apply the current model
			vct.getAspectRating(m_model.m_beta);
			for(int i=0; i<vct.m_alpha.length; i++)
				orating += vct.m_alpha[i] * vct.m_pred[i];
			likelihood += orating*orating;
		}
		return likelihood;
	}
	
	//\sum_d\pi\sum_i\alpha_{di}(\S_{di}-r_d)^2
	protected double getAuxDataLikelihood(){
		double likelihood = 0, orating;
						
		// part I of objective function: data likelihood
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			orating = vct.m_ratings[0];
			for(int i=0; i<vct.m_alpha.length; i++)
				likelihood += vct.m_alpha[i] * (vct.m_pred[i] - orating)*(vct.m_pred[i] - orating);
		}
		return PI * likelihood;
	}

	protected double getBetaObjGradient(){
		double likelihood = 0, aux_likelihood = 0, orating, diff, oRate;
		int vSize = m_model.m_v + 1, offset;
		SpaVector sVct;
		
		// initialize the structure
		Arrays.fill(m_g_beta, 0);
				
		// part I of objective function: data likelihood
		for(Vector4Review vct:m_collection){
			if (vct.m_4train==false)
				continue; // do not touch testing cases
			
			oRate = vct.m_ratings[0];
			orating = -oRate;
			
			//apply the current model
			vct.getAspectRating(m_beta, vSize);
			for(int i=0; i<m_model.m_k; i++)
				orating += vct.m_alpha[i] * vct.m_pred[i];			
			
			likelihood += orating*orating;
			orating /= m_model.m_delta; // in order to get consistency between aux-likelihood
			
			offset = 0;
			for(int i=0; i<m_model.m_k; i++){
				aux_likelihood += vct.m_alpha[i]* (vct.m_pred[i]-oRate)*(vct.m_pred[i]-oRate);
				if (RatingRegression.SCORE_SQUARE)
					diff = vct.m_alpha[i]*(orating + PI*(vct.m_pred[i]-oRate)) * vct.m_pred_cache[i];
				else
					diff = vct.m_alpha[i]*(orating + PI*(vct.m_pred[i]-oRate)) * vct.m_pred[i];
				
				sVct = vct.m_aspectV[i];
				m_g_beta[offset] += diff;//first for bias term
				for(int j=0; j<sVct.m_index.length; j++)
					m_g_beta[offset + sVct.m_index[j]] += diff*sVct.m_value[j];
				offset += vSize;//move to next aspect
			}
		}
		
		double reg = 0;
		for(int i=0; i<m_beta.length; i++)
		{
			m_g_beta[i] += m_lambda*m_beta[i];
			reg += m_beta[i]*m_beta[i];
		}
		
		return 0.5*(likelihood/m_model.m_delta + PI*aux_likelihood + m_lambda*reg);
	}
	
	protected double ml_beta() throws ExceptionWithIflag{
		double f = 0;
		int iprint [] = {-1,0}, iflag[] = {0}, icall = 0, n = (1+m_model.m_v)*m_model.m_k, m = 5;
		
		for(int i=0; i<m_model.m_k; i++)//set up the starting point
			System.arraycopy(m_model.m_beta[i], 0, m_beta, i*(m_model.m_v+1), m_model.m_v+1);
		
		Arrays.fill(m_diag_beta, 0);
		do {
			if (icall%1000==0)
				System.out.print(".");//keep track of beta update
			f = getBetaObjGradient();//to be minimized
			LBFGS.lbfgs ( n , m , m_beta , f , m_g_beta , false , m_diag_beta , iprint , m_betaTol , 1e-20 , iflag );
		} while ( iflag[0] != 0 && ++icall <= m_betaStep );

		System.out.print(icall + "\t");
		for(int i=0; i<m_model.m_k; i++)
			System.arraycopy(m_beta, i*(m_model.m_v+1), m_model.m_beta[i], 0, m_model.m_v+1);
		return f;
	}
	
	public void EM_est(String filename, int maxIter, double converge){
		int iter = 0, alpha_exp = 0, alpha_cov = 0;
		double tag, diff = 10, likelihood = 0, old_likelihood = init(LoadVectors(filename));
				
		System.out.println("[Info]Step\toMSE\taMSE\taCorr\tiCorr\tcov(a)\texp(a)\tobj\tconverge");
		while(iter<Math.min(8, maxIter) || (iter<maxIter && diff>converge)){
			alpha_exp = 0;
			alpha_cov = 0;
			
			//E-step
			for(Vector4Review vct:m_collection){
				if (vct.m_4train){
					tag = EStep(vct);
					if (tag==-1) // failed to converge
						alpha_cov ++;
					else if (tag==-2) // failed with exceptions
						alpha_exp ++;
				}					
			}			
			System.out.print(iter + "\t");//sign of finishing E-step
			
			//M-step
			likelihood = MStep(iter%4==3);//updating \Sigma too often will hamper \hat\alpha convergence		
			
			evaluateAspect();// evaluating in the testing cases
			diff = (old_likelihood-likelihood)/old_likelihood;
			old_likelihood = likelihood;
			System.out.println(String.format("\t%d\t%d\t%.3f\t%.3f", alpha_cov, alpha_exp, likelihood, diff));
			iter++;
		}
		
		try {
			m_trace.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void SaveModel(String filename){
		m_model.Save2File(filename);
	}
	
	public static void main(String[] args) {
		LRR model = new LRR(500, 1e-2, 5000, 1e-2, 2.0);//
		model.EM_est("Data/Vectors/Vector_CHI_4000.dat", 10, 1e-4);
		model.SavePrediction("Data/Results/prediction.dat");
		model.SaveModel("Data/Model/model_hotel.dat");
	}
}
