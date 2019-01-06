import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;

public class Center {
    public Center(File text_to_analyze) throws IOException, ClassNotFoundException {
        String toanalyze = new String(Files.readAllBytes(text_to_analyze.toPath()));
        String[] words = textReduction(toanalyze);
        Word2Vector vecmod = new Word2Vector();
        vecmod.load();
        Vector[] vecs = new Vector[words.length];
        for(int i =0; i<words.length;i++){
            vecs[i] = vecmod.vectorize(words[i]);
        }
        Vector fulltext = flatten(vecs);
        //deserialize all the networks we need 
        /*
        RecurrentNetwork sad = RecurrentNetwork.deserialize("networks/sad");
        RecurrentNetwork happy = RecurrentNetwork.deserialize("networks/happy");
        RecurrentNetwork apathetic = RecurrentNetwork.deserialize("networks/apathetic");
        */
        NeuralNetwork moodClassifier = NeuralNetwork.deserialize("networks/mood");
        Vector mood = moodClassifier.forwardPass(fulltext);
        double happy = mood.values[0];
        double sad = mood.values[1];
        double apathetic = mood.values[2];
        Vector[] essay;
        if(happy>sad && happy>apathetic){
            RecurrentNetwork writer = RecurrentNetwork.deserialize("networks/happy");
            essay = writer.forwardPropogate(vecs);
        }else if(sad>apathetic&&sad>happy){
            RecurrentNetwork writer = RecurrentNetwork.deserialize("networks/sad");
            essay = writer.forwardPropogate(vecs);
        }else{
            RecurrentNetwork writer = RecurrentNetwork.deserialize("networks/apathetic"); 
            essay = writer.forwardPropogate(vecs);
        }
        System.out.println(vecmod.toWords(essay));
    }
    public String[] textReduction(String input){
        String[] words = input.split(" ");
        final String[] UNIMPORTANT = {"my","it","there","at","me","i","the","is","are","or","that","those","by","of","have","them","for","how","on","both","such","as","was","will","be"};
        final ArrayList<String> UNIMPORTANT_ARRAYLIST = new ArrayList<>(Arrays.asList(UNIMPORTANT));
        ArrayList<String> temp = new ArrayList<>();
            for(String word : words){
                if(!UNIMPORTANT_ARRAYLIST.contains(word)){
                    temp.add(word);
                    if(temp.size()==50)break;
                }
            }
        return temp.toArray(new String[temp.size()]);
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
}