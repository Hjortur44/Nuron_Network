import java.util.List;

public class NuronNetwork {
	
	public NuronNetwork() {

	}

	public double node(List<Double> input, List<Double> weight, double bias) {
		double sum = 0.0;

		for(int i = 0; i < input.size(); i++) {
			sum += input.get(i) * weight.get(i);
		}
		
		return sigmoid(sum + bias);
	}
	
	// Square Sum Error function.
	public double error(List<Double> real, List<Double> expected) {
		if(real.size() != expected.size()) return -1.0;
		
		double sum = 0.0;
		
		for(int i = 0; i < real.size(); i++) {
			double dif = (real.get(i) - expected.get(i));
			sum += dif * dif;
		}
		
		return sum;		
	}	
	
	// Sigmoid.
	public double sigmoid(double x) {
		return (1.0 / (1.0 + Math.pow(Math.E, (-x))));		
	}
	
	// Derivative of Sigmoid.
	public double sigmoidDerivative(double input, double weight) {
		double s = sigmoid(input * weight);
		return s * (1 - s);
	}
	
	// Derivative of the cost function (Square Sum Error).
	public double errorDerivative(double real, double expected) {
		return 2 * (real - expected);
	}
	
	// Derivative of the input.
	public  double inputDerivative(double input) {
		return input;
	}

	private boolean validate(int a, int b) {
		return a == b;
	}
}
