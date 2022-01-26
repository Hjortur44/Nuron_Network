import java.util.ArrayList;
import java.util.List;

public class NuronNetwork {
	
	public NuronNetwork() {

	}

	public double sigmoidNode(List<Double> inputs, List<Double> weights, double bias) {
		return sigmoid(calc(inputs, weights) + bias);
	}
	
	public double reluNode(List<Double> inputs, List<Double> weights, double bias) {		
		return relu(calc(inputs, weights) + bias);
	}
	
	// Mean Square Error.
	public double error(List<Double> real, List<Double> expected) {
		if(real.size() != expected.size()) return -1.0;
		
		double sum = 0.0;
		
		for(int i = 0; i < real.size(); i++) {
			double dif = (real.get(i) - expected.get(i));
			sum += dif * dif;
		}
		
		return sum / real.size();
	}
	
	// Sum += Inputs * Weights calculations.
	private double calc(List<Double> input, List<Double> weight) {
		double sum = 0.0;

		for(int i = 0; i < input.size(); i++) {
			sum += input.get(i) * weight.get(i);
		}
		
		return sum;
	}
	
	// RELU
	private double relu(double x) {
		return (x < 0)? 0 : x;
	}

	// Sigmoid.
	private double sigmoid(double x) {
		return (1.0 / (1.0 + Math.pow(Math.E, (-x))));		
	}
	
	// Derivative of Sigmoid.
	public double sigmoidDerivative(double input, double weight) {
		double s = sigmoid(input * weight);
		return s * (1 - s);
	}
	
	// Derivative of the cost function (Mean Square Error).
	public double errorDerivative(double real, double expected) {
		return 2 * (real - expected);
	}

	private boolean validate(int a, int b) {
		return a == b;
	}
}
