import java.io.FileWriter;
import java.io.IOException;

public class DataCreater {

	private int inputSize, weightSize, expectedSize, biasSize;
	
	public DataCreater(int inputSize, int weightSize, int expectedSize, int biasSize) {	
		this.inputSize = inputSize;
		this.weightSize = weightSize;
		this.expectedSize = expectedSize;
		this.biasSize = biasSize;
	}
	
	public void create() {
		double[] b = new double [biasSize];
		double[] w = new double [weightSize];
		double[] in = new double[inputSize];
		double[] e = new double[expectedSize];
	
		for(int i = 0; i < inputSize; i++) {
			in[i] = Math.random();			
		}
		
		for(int i = 0; i < expectedSize; i++) {
			e[i] = Math.random();
		}	
		
		for(int i = 0; i < weightSize; i++) {
			w[i] = Math.random();
		}
		
		for(int i = 0; i < biasSize; i++) {
			b[i] = Math.random();
		}

		writeToFile("weights", w);
		writeToFile("inputs", in);
		writeToFile("expected", e);
		writeToFile("biases", b);
	}
	
	private void writeToFile(String name, double[] v) {
		try {
			FileWriter file = new FileWriter(name + ".txt");
			
			for(double val : v) file.append(val + "\n");			
			file.close();
			
			System.out.println("Wrote data to the file: " + name + ".txt");
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}
