import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Driver {
	private static NuronNetwork net = new NuronNetwork();
	private static List<Double> biases = new ArrayList<>();
	private static List<Double> expected = new ArrayList<>();
	private static List<Double> weights = new ArrayList<>();
	private static List<Double> activations = new ArrayList<>();
	private static List<Double> inputs = new ArrayList<>();
	private static List<Double> outputs = new ArrayList<>();
	
	private static void dataWrite(int inputSize, int weightSize, int expectedSize, int biasSize) {
		DataCreater data = new DataCreater(inputSize, weightSize, expectedSize, biasSize);
		data.create();
	}
	
	private static void dataRead() {
		int i = 0;
		
		try {
			File file = new File("expected.txt");
			Scanner reader = new Scanner(file);
			i = 0;
			
			while(reader.hasNext()) {
				expected.add(Double.parseDouble(reader.next()));				
			}
			
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		try {
			File file = new File("inputs.txt");
			Scanner reader = new Scanner(file);
			i = 0;
			
			while(reader.hasNext()) {
				inputs.add(Double.parseDouble(reader.next()));				
			}
			
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	
		try {
			File file = new File("weights.txt");
			Scanner reader = new Scanner(file);
			i = 0;
			
			while(reader.hasNext()) {
				weights.add(Double.parseDouble(reader.next()));			
			}
			
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		try {
			File file = new File("biases.txt");
			Scanner reader = new Scanner(file);
			i = 0;
			
			while(reader.hasNext()) {
				biases.add(Double.parseDouble(reader.next()));			
			}
			
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}		
	}

	//------------------------------------------------------------------------
	
	// The shape of the hidden layers.
	// The length of this array is the depth of the network,
	// while individual numbers describe the amount of nodes for each layer.
	private static void network(int[] hiddenLayerShapes, int outputSize) {
	
		int weightCounter = 0;
		
		// Inputs * Weights = Activations
		int shape = hiddenLayerShapes[0];
		
		for(int i = 0; i < shape; i++) {
			List<Double> w = new ArrayList<>();

			for(int j = i; j < (inputs.size() * shape); j += shape) {
				w.add(weights.get(j));
			}

			weightCounter += w.size();
			double n = net.sigmoidNode(inputs, w, biases.get(0));
			activations.add(n);			
		}		
		
		// Activations(x-1) * Weights = Activations(x)
		for(int i = 1; i < hiddenLayerShapes.length; i++) {
			int prevShape = hiddenLayerShapes[i - 1];
			int currShape = hiddenLayerShapes[i];
			List<Double> actSub = activations.subList(activations.size() - prevShape, activations.size());
			List<Double> tmp = new ArrayList<>();
			
			for(int j = 0; j < currShape; j++) {
				List<Double> w = new ArrayList<>();
				
				for(int k = j; k < prevShape * currShape; k += currShape) {
					w.add(weights.get(k + weightCounter));
				}
	
				weightCounter += prevShape;
				double n = net.sigmoidNode(actSub, w, biases.get(i));
				tmp.add(n);				
			}
			
			activations.addAll(tmp);
		}

		// Activations * Weights = Outputs
		int lastLayer = hiddenLayerShapes[hiddenLayerShapes.length - 1];
		List<Double> actSub = activations.subList(activations.size() - lastLayer, activations.size());

		for(int i = 0; i < outputSize; i++) {	
			for(int j = i; j < actSub.size(); j += outputSize) {
				List<Double> w = new ArrayList<>();
				
				for(int k = weightCounter + j; k < weights.size(); k += outputSize) {
					w.add(weights.get(k));	
				}
	
				double n = net.reluNode(actSub, w, 0.0);
				outputs.add(n);
			}
		}
	}
	
	//------------------------------------------------------------------------
	public static void main(String[] args) {
		int[] hiddenLayerShapes = {5, 1};
		int length = hiddenLayerShapes.length;
		int inputSize = 1;
		int expectedSize = hiddenLayerShapes[length - 1];
		int biasSize = length;
		int outputSize = expectedSize;
		int weightSize = inputSize * hiddenLayerShapes[0];
		double learning = 0.1;
		
		for(int i = 1; i < length; i++) {
			weightSize += hiddenLayerShapes[i - 1] * hiddenLayerShapes[i];
		}

		weightSize += expectedSize * outputSize;

		//dataWrite(inputSize, weightSize, expectedSize, biasSize); 
		dataRead();
		network(hiddenLayerShapes, outputSize);
		for(double d : weights) System.out.println("Pre_Weights: " + d);
		for(double d : outputs) System.out.println("Pre_Outputs: " + d);
		
		for(int i = 0; i < 4000; i++) {
			double oldWeight1 = weights.remove(1);
			double cost = net.errorDerivative(outputs.remove(0), expected.get(0));
			double sig = net.sigmoidDerivative(activations.remove(0), oldWeight1);
	
			double newWeight1 = oldWeight1 - (learning * oldWeight1 * sig * cost);
			weights.add(newWeight1);

			activations.clear();
			network(hiddenLayerShapes, outputSize);
			
			double oldWeight0 = weights.remove(0);
			double cost2 = net.errorDerivative(outputs.remove(0), expected.get(0));	
			double sig2 = net.sigmoidDerivative(inputs.get(0), oldWeight0);
			
			double newWeight0 = oldWeight0 - (learning * oldWeight0  * sig2 * cost2);

			weights.add(0, newWeight0);
			activations.clear();
			network(hiddenLayerShapes, outputSize);
			
			
			// Stop guard.
			if(outputs.get(0) < expected.get(0) && outputs.get(0) / expected.get(0) > 0.99) {
				System.out.println("Stopped: " + i);
				break;
			}
			
		}

		for(double d : expected) System.out.println("Expected: " + d);
		for(double d : weights) System.out.println("Post_Weights: " + d);
		for(double d : outputs) System.out.println("Post_Outputs: " + d);
	}
}
