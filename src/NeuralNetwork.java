
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;

public class NeuralNetwork implements Serializable{
    private static final long serialVersionUID = 1L;
    public static int[] structure = { 700, 20, 4, 3 };
ArrayList<Layer> layers;    
public final static double LEARNING_RATE = .1;
public NeuralNetwork(int inputsize){
        layers = new ArrayList<>();
        for(int i =0; i<structure.length; i++){
            layers.add(
            Layer.random(structure[i],
            i==0?inputsize:structure[i-1])
            );
        }
}
public Vector[] biasZeroes(){
Vector[] ret = new Vector[layers.size()];
    for(int i = 0; i<layers.size(); i++){
    double[] toad = new double[layers.get(i).bias.values.length];
    for(int j=0; j<toad.length; j++){
        toad[j] = 0;
    }
    ret[i] = new Vector(toad);
}
return ret;
}
public Matrix[] weightsZeroes(){
    Matrix[] ret = new Matrix[layers.size()];
    for(int i = 0; i<layers.size(); i++){
    double[][] toad = new double[layers.get(i).weights.M][layers.get(i).weights.N];
    for(int j=0; j<toad.length; j++){
        for(int k = 0; k<toad[j].length;k++)
        toad[j][k] = 0;
    }
    ret[i] = new Matrix(toad);
}
return ret;
}
public void train(Vector[] inputs, Vector[] outputs){
Vector[] biasdeltas = biasZeroes();
Matrix[] weightdeltas = weightsZeroes();
for(int i = 0; i<inputs.length; i++){
    for(int j=0;j<layers.size();j++){
        biasdeltas[j] = biasdeltas[j].toMatrix().plus(backwardpassbias(j, inputs[i], outputs[i]).toMatrix()).toVector();
        weightdeltas[j] = weightdeltas[j].plus(backwardpassweights(j,inputs[i],outputs[i]));
    }
}
for(int i=0; i<biasdeltas.length;i++){
    biasdeltas[i].scale(LEARNING_RATE*1/inputs.length);
    weightdeltas[i].scale(LEARNING_RATE*1/inputs.length);
}
//Actually do gradient descent
for(int i=0; i<layers.size(); i++){
    layers.get(i).weights = layers.get(i).weights.minus(weightdeltas[i]);
    layers.get(i).bias = layers.get(i).bias.toMatrix().minus(biasdeltas[i].toMatrix()).toVector();
}
}
public Vector backwardpassbias(int layer,Vector input,Vector output){
    Vector yhat = forwardPass(input);
    Vector cur = null;
    for(int i = layers.size()-1; i>=layer;i--){
        Vector deltas = layers.get(i).deltas;
        if(i!=layer){
            if(cur==null){
                cur = deltas.toMatrix().times(layers.get(i).weights.transpose()).toVector();
            }else{
                Matrix matrtemp = deltas.toMatrix().times(layers.get(i).weights.transpose());
                matrtemp.print();
                cur.print();
                cur = matrtemp.toVector().hadamardProduct(cur);
            }
        }else{
            cur = cur.hadamardProduct(deltas);
        }
    }
    return output.toMatrix().minus(yhat.toMatrix()).toVector().hadamardProduct(cur);
}
public Matrix backwardpassweights(int layer,Vector input,Vector output){
    Vector yhat = forwardPass(input);
    Vector cur = null;
    for(int i = layers.size()-1; i>=layer;i--){
        Vector deltas = layers.get(i).deltas;
        if(i!=layer){
            if(cur==null){
                cur = deltas.toMatrix().times(layers.get(i).weights.transpose()).toVector();
            }else{
                cur = deltas.toMatrix().times(layers.get(i).weights.transpose()).toVector().hadamardProduct(cur);
            }
        }else{
            /*
            we just treat each output as a separate thing being optimized for
            and then minimize the average
            cuz idk linear algebra too well ngl
            */
            Matrix retMatrix = weightsZeroes()[layer];
            for(int j = 0; j<output.values.length; j++){
                retMatrix = retMatrix.plus(cur.toMatrix().times(layers.get(layer).savedinput.toMatrixVertical()).scale(output.values[j]-yhat.values[j]));
            }
            return retMatrix.scale(1/output.values.length);
        }
    }
    //this is never reached
    return null;
}
public Vector forwardPass(Vector input){
    Vector cur = input;
    for(Layer l : layers){
        cur = l.forwardPass(cur);
    }
    return cur;
}
public double score(Vector[] inputs,Vector[] outputs){
    /*
    this is NOT the function being optimized it's just a quick representation 
    so you can get a rough feeling of network performance
    */
    double error = 0;
    for(int i=0; i<inputs.length; i++){
        Vector yhat = forwardPass(inputs[i]);
        error+=outputs[i].toMatrix().minus(yhat.toMatrix()).toVector().sum();
    }
    return error;
}
public static void main(String[] args){
     Vector input = new Vector(new double[]{3,2});
     NeuralNetwork n = new NeuralNetwork(2);
     n.forwardPass(input).print();
     System.out.println("forward pass is functional");
     Vector one = new Vector(new double[]{1,0});
     Vector two = new Vector(new double[]{0,1});
     Vector three = new Vector(new double[]{0,0});
     Vector four = new Vector(new double[]{1,1});
     Vector five = new Vector(new double[]{1});
     Vector six = new Vector(new double[]{1});
     Vector seven = new Vector(new double[]{0});
     Vector eight = new Vector(new double[]{0});
     for(int i =0; i<8; i++){
         System.out.println("epoch "+i+" of 8");
         System.out.println("score: "+n.score(new Vector[]{one,two,three,four},new Vector[]{five,six,seven,eight}));
         n.train(new Vector[]{one,two,three,four},new Vector[]{five,six,seven,eight});
    }

    }
public static NeuralNetwork deserialize(String path) throws ClassNotFoundException, IOException {
	// Reading the object from a file 
    FileInputStream file = new FileInputStream(path); 
    ObjectInputStream in = new ObjectInputStream(file); 
      
    // Method for deserialization of object 
    NeuralNetwork ret = (NeuralNetwork)in.readObject(); 
      
    in.close(); 
    file.close(); 
    return ret;
}
public void serialize(String path) {
    try{    
        //Saving of object in a file 
        FileOutputStream file = new FileOutputStream(path); 
        ObjectOutputStream out = new ObjectOutputStream(file); 
          
        // Method for serialization of object 
        out.writeObject(this); 
          
        out.close(); 
        file.close(); 
          
        return;

    }catch(IOException ex) { 
        System.out.println("IOException is caught"); 
    } 
}
}
