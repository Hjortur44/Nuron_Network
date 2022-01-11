import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class Driver {
	
	public static void main(String[] args) {
		int row = 4, col = 2, exp = 2, b = 1;
		//DataCreater data = new DataCreater(row, col, exp, b);
		//data.create();

		double[] biases = new double[b];
		double[] expected = new double[exp];
		double[] weights = new double[row * col];
		double[] inputs = new double[row];		
		int i = 0;
		
		try {
			File file = new File("expected.txt");
			Scanner reader = new Scanner(file);
			i = 0;
			
			while(reader.hasNext()) {
				expected[i++] = Double.parseDouble(reader.next());				
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
				inputs[i++]= Double.parseDouble(reader.next());				
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
				weights[i++] = Double.parseDouble(reader.next());				
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
				biases[i++] = Double.parseDouble(reader.next());				
			}
			
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		//--------------------------------------------------------------------------------------------


		NuronNetwork forward = new NuronNetwork();
		int[] hiddenLayerShapes = {2};
		
		for(int hidden = 0; hidden < hiddenLayerShapes.length; hidden++) {
			double[] nodes = new double[hiddenLayerShapes[hidden]];
			
			for(int j = 0; j < col; j++) {
				double[] w = new double[row];
				i = 0;
				
				for(int k = j; k < weights.length; k += col) {
					w[i++] = weights[k];
				}
			
				nodes[j] = forward.node(inputs, w, biases[hidden]);	
			}

			inputs = nodes;
		}
		
		
		for(int j = 0; j < inputs.length; j++) {
			System.out.println(inputs[j]);
		}
	
		double err = forward.error(inputs, expected);
		System.out.println(err);
		
	}
}
