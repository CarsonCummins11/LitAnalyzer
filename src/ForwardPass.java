import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;

public class ForwardPass {

	public ForwardPass() throws IOException, ClassNotFoundException {
		
		String text = new String(Files.readAllBytes(new File("poem").toPath()));
		String[] words = textReduction(text);
		for(String wor : words) {
			System.out.print(wor+" ");
		}
		System.out.println();
		NeuralNetwork MoodClass = new NeuralNetwork(15000);
		Word2Vector vecmod = new Word2Vector();
        vecmod.load();
        
		Vector[] vecs = new Vector[words.length];
		for(int i =0; i<words.length;i++){
            vecs[i] = vecmod.vectorize(words[i]);
        }
		vecs[2].print();
        //serialize(vecs);
        
        //Vector[] vecs = deserialize("vecs");
        Vector fulltext = flatten(vecs);
        Vector mood = MoodClass.forwardPass(fulltext);
        System.out.println("mood");
        mood.print();
        RecurrentNetwork writer = new RecurrentNetwork(50,300);
        Vector[] analysis = writer.forwardPropogate(vecs);

        System.out.println(vecmod.toWords(analysis));
     
	}
	private void serialize(Vector[] vecs) {
		try{    
	        //Saving of object in a file 
	        FileOutputStream file = new FileOutputStream("vecs"); 
	        ObjectOutputStream out = new ObjectOutputStream(file); 
	          
	        // Method for serialization of object 
	        out.writeObject(vecs); 
	          
	        out.close(); 
	        file.close(); 
	          
	        return;

	    }catch(IOException ex) { 
	        System.out.println("IOException is caught"); 
	    } 
		
	}
	public static Vector[] deserialize(String path) throws ClassNotFoundException, IOException {
		// Reading the object from a file 
	    FileInputStream file = new FileInputStream(path); 
	    ObjectInputStream in = new ObjectInputStream(file); 
	      
	    // Method for deserialization of object 
	    Vector[] ret = (Vector[])in.readObject(); 
	      
	    in.close(); 
	    file.close(); 
	    return ret;
	}
	//converts vector array to single vector
	public Vector flatten(Vector[] inputs){
	        ArrayList<Double> rr = new ArrayList<>();
	        for(Vector vv : inputs){
	            for (double d : vv.values){
	                rr.add(d);
	            }
	        }
	        double[] d = new double[rr.size()];
	        for(int i=0; i<d.length;i++){
	            d[i] = rr.get(i);
	        }
	        return new Vector(d);

	}
	public static void main(String[] args) {
		try {
			new ForwardPass();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	 public String[] textReduction(String input){
		 input = input.replaceAll("\n"," ");
		 input = input.replaceAll("\r"," ");
		 input = input.replace(".","");
		 input = input.replaceAll(",","");
		 input = input.replace("—"," ");
		 input = input.replace("-","" );
		 input = input.trim();
	        String[] words = input.split(" ");
	        final String[] UNIMPORTANT = {"a","I","and","my","it","there","at","me","i","the","is","are","or","that","those","by","of","have","them","for","how","on","both","such","as","was","will","be"};
	        final ArrayList<String> UNIMPORTANT_ARRAYLIST = new ArrayList<>(Arrays.asList(UNIMPORTANT));
	        ArrayList<String> temp = new ArrayList<>();
	            for(String word : words){
	                if(!UNIMPORTANT_ARRAYLIST.contains(word)){
	                    temp.add(word);
	                    if(temp.size()==50)break;
	                }
	            }
	        if(temp.size()!=50) {
	        	temp = new ArrayList<String>();
	        	for(int i=0; i<50; i++) {
	        		temp.add(words[i]);
	        	}
	        }
	        return temp.toArray(new String[temp.size()]);
	    }
}
