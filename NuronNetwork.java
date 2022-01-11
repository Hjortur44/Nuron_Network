
public class NuronNetwork {
	
	public NuronNetwork() {			

	}	

	public double node(double[] input, double[] weight, double bias) {
		double sum = 0.0;
		
		for(int i = 0; i < input.length; i++) {
			sum += input[i] * weight[i];
		}
		
		return activation(sum + bias);
	}
	
	// Mean Sqare Error function.
	public double error(double[] real, double[] expected) {
		if(real.length != expected.length) return -1.0;
		
		double sum = 0.0;
		
		for(int i = 0; i < real.length; i++) {
			sum += (real[i] - expected[i]) * (real[i] - expected[i]);
		}
		
		return sum / real.length;		
	}
	
	// Sigmoid.
	private double activation(double x) {
		return (1.0 / (1.0 + Math.pow(Math.E, (-x))));		
	}	

	private boolean validate(int a, int b) {
		return a == b;
	}
}
