

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;

public class Training {
    public static final int HAPPY = 1;
    public static final int APATHETIC = 0;
    public static final int SAD = -1;
public Training(String[][] datapaths) throws IOException{
ArrayList<Vector> classifierInputs = new ArrayList<>();
ArrayList<Vector> classifierOutputs = new ArrayList<>();
ArrayList<Vector[]> writerInputs = new ArrayList<>();
ArrayList<Vector[]> writerOutputs = new ArrayList<>();
for(String[] data : datapaths){
    String inputpath = data[0];
    String outputpath = data[1];
    String toanalyze = new String(Files.readAllBytes(new File(inputpath).toPath()));
    String[] words = textReduction(toanalyze);
    Word2Vector vecmod = new Word2Vector();
    vecmod.load();
    Vector[] vecs = new Vector[words.length];
    for(int i =0; i<words.length;i++){
        vecs[i] = vecmod.vectorize(words[i]);
    }
    String toanalyze_out = new String(Files.readAllBytes(new File(outputpath).toPath()));
    toanalyze_out = getFirstNWords(toanalyze_out,50);
    String[] words_out = toanalyze_out.split(" ");
    Vector[] vecs_out = new Vector[words.length];
    for(int i =0; i<words_out.length;i++){
        vecs_out[i] = vecmod.vectorize(words_out[i]);
    }
    writerInputs.add(vecs);
    writerOutputs.add(vecs_out);
    int mood = Integer.parseInt(data[2]);
    if(mood ==0){
    classifierOutputs.add(new Vector(new double[]{0,0,1}));
    }else if(mood ==-1){
        classifierOutputs.add(new Vector(new double[]{0,1,0}));
    }else{
        classifierOutputs.add(new Vector(new double[]{1,0,0}));
    }
}
for(Vector[] v:writerInputs){
classifierInputs.add(flatten(v));
}
Vector[] classTrainingInputs = classifierInputs.toArray(new Vector[classifierInputs.size()]);
Vector[] classTrainingOutputs = classifierOutputs.toArray(new Vector[classifierOutputs.size()]);

ArrayList<Vector[]> happyWriting = new ArrayList<>();
ArrayList<Vector[]> sadWriting = new ArrayList<>();
ArrayList<Vector[]> apatheticWriting = new ArrayList<>();
ArrayList<Vector[]> happyAnalysis = new ArrayList<>();
ArrayList<Vector[]> sadAnalysis = new ArrayList<>();
ArrayList<Vector[]> apatheticAnalysis = new ArrayList<>();
for(int i = 0; i< writerInputs.size(); i++){
if(classifierOutputs.get(i).values[2]==1){
    apatheticWriting.add(writerInputs.get(i));
    apatheticAnalysis.add(writerOutputs.get(i));
}else if(classifierOutputs.get(i).values[1]==1){
    sadWriting.add(writerInputs.get(i));
    sadAnalysis.add(writerOutputs.get(i));
}else{
    happyWriting.add(writerInputs.get(i));
    happyAnalysis.add(writerOutputs.get(i));
}
}
RecurrentNetwork happyNetwork = new RecurrentNetwork(50, 25);
RecurrentNetwork apatheticNetwork = new RecurrentNetwork(50,25);
RecurrentNetwork sadNetwork = new RecurrentNetwork(50,25);
NeuralNetwork toneNetwork = new NeuralNetwork(50);
toneNetwork.train(classTrainingInputs, classTrainingOutputs);
happyNetwork.train(
    happyWriting.toArray(new Vector[happyWriting.size()][]),
    happyAnalysis.toArray(new Vector[happyWriting.size()][])
);
sadNetwork.train(
    sadWriting.toArray(new Vector[sadWriting.size()][]),
    sadAnalysis.toArray(new Vector[sadAnalysis.size()][])
);
apatheticNetwork.train(
    apatheticWriting.toArray(new Vector[apatheticWriting.size()][]), 
    apatheticAnalysis.toArray(new Vector[apatheticWriting.size()][])
);
File networkFolder = new File("networks");
if(!networkFolder.exists())networkFolder.mkdir();
sadNetwork.serialize("networks/sad");
happyNetwork.serialize("networks/happy");
apatheticNetwork.serialize("networks/apathetic");
toneNetwork.serialize("networks/mood");

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
public String getFirstNWords(String s,int n){
    int wordcount =0;
    int at = 0;
    s.replaceAll("\n", " ");
    s.replaceAll("\t", " ");
    s.replaceAll("\r", " ");
    s = s.trim();
    String ret = "";
    while(wordcount<n){
        char c = s.charAt(at);
        if(c==' '){
            wordcount++;
        }
        ret+=c;
        at++;
    }
    return ret;
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
}