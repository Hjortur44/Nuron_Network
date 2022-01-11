import java.io.FileWriter;
import java.io.IOException;

public class DataCreater {

	private int row, col, exp, bias;
	
	public DataCreater(int row, int col, int exp, int bias) {
		this.row = row;
		this.col = col;
		this.exp = exp;
		this.bias = bias;
	}
	
	public void create() {
		double[] b = new double [bias];
		double[] w = new double [row * col];
		double[] v = new double[row];
		double[] e = new double[exp];
	
		for(int i = 0; i < v.length; i++) {
			v[i] = Math.random();			
		}
		
		for(int i = 0; i < exp; i++) {
			e[i] = Math.random();
		}	
		
		for(int i = 0; i < w.length; i++) {
			w[i] = Math.random();
		}
		
		for(int i = 0; i < b.length; i++) {
			b[i] = Math.random();
		}

		writeToFile("weights", w);
		writeToFile("inputs", v);
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
