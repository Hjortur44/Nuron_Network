import java.util.ArrayList;
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
	
	// Mean Square Error function.
	public double error(List<Double> real, List<Double> expected) {
		if(real.size() != expected.size()) return -1.0;
		
		double sum = 0.0;
		
		for(int i = 0; i < real.size(); i++) {
			double dif = (real.get(i) - expected.get(i)) * (real.get(i) - expected.get(i));

			sum += dif;
		}
		
		return sum / real.size();
	}
	
	public double weightAjustment(double prevWeight, double prevActivation, double input, double output, double expected, double learningRate) {
	
		double err = errorDerivative(output, expected);
		double sig = sigmoidDerivative(prevActivation, prevWeight);
		double in = input;

		double w = learningRate * in * sig * err;
		return prevWeight - w;		
	}
	
	// Sigmoid.
	private double sigmoid(double x) {
		return (1.0 / (1.0 + Math.pow(Math.E, (-x))));		
	}
	
	// Derivative of Sigmoid.
	private double sigmoidDerivative(double input, double weight) {
		double s = sigmoid(input * weight);
		return s * (1 - s);
	}
	
	// Derivative of the cost function (Mean Square Error).
	private double errorDerivative(double real, double expected) {
		return 2 * (real - expected);
	}
	
	// Derivative of the input.
	private  double inputDerivative(double input) {
		return input;
	}

	private boolean validate(int a, int b) {
		return a == b;
	}
}
