package lara;

public class SpaVector {
	public int[] m_index;//index must start from 1
	public double[] m_value;
	
	public SpaVector(String[] container){
		m_index = new int[container.length];
		m_value = new double[container.length];
		
		int pos;
		for(int i=0; i<container.length; i++){
			pos = container[i].indexOf(':');
			m_index[i] = 1 + Integer.valueOf(container[i].substring(0,pos));
			m_value[i] = Double.valueOf(container[i].substring(pos+1));
		}
	}
	
	public double L1Norm(){
		double sum = 0;
		for(double v:m_value)
			sum += Math.abs(v);
		return sum;
	}
	
	public void normalize(double norm){
		for(int i=0; i<m_value.length; i++)
			m_value[i] /= norm;
	}
	
	public int getLength(){
		int i = m_index.length;
		return m_index[i-1];
	}
	
	public double dotProduct(double[] weight){
		double sum = weight[0]; // the bias term
		for(int i=0; i<m_index.length; i++)
			sum += m_value[i] * weight[m_index[i]];
		return sum;
	}
	
	public double dotProduct(double[] weight, int offset){
		double sum = weight[offset]; // the bias term
		for(int j=0; j<m_index.length; j++)
			sum += m_value[j] * weight[offset + m_index[j]]; // index starts from one
		return sum;
	}
}
