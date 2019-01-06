
import java.io.File;


import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;



public class Word2Vector {
    public Word2Vec model;
 
    public Word2Vector(){

    }
    
    public void load() {
        File  modFile = new File("GoogleNews-vectors-negative300.bin.gz");
        System.out.println("starting word2vec load");
        model = WordVectorSerializer.readWord2VecModel(modFile);      
        System.out.println("finished word2vec load");
    }
    public Vector vectorize(String word){
        return new Vector(model.getWordVector(word));
    }
    public String vec2word(Vector v){
       return model.wordsNearest(v.toINDArray(),1).iterator().next();
    }
    public Vector sigmoid(Vector in){
        double[] ret = new double[in.values.length];
        for(int i=0;i<in.values.length;i++){
            ret[i] = 1/(1+Math.exp(-in.values[i]));
        }
        return new Vector(ret);
    }
	public String toWords(Vector[] essay) {
		String ret = "";
		for (Vector v : essay) {
			v= sigmoid(v);
			ret+= vec2word(v)+" ";
		}
		return ret;
	}
}