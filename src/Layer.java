
import java.io.Serializable;

public class Layer implements Serializable{
    private static final long serialVersionUID = 1L;
    Matrix weights;
    Vector bias;
    Vector deltas;
    Vector savedinput;
    public static Layer random(int size, int previousSize){
        Layer ret = new Layer();
        ret.bias = Vector.random(size);
        ret.weights = Matrix.random(previousSize,size);
        return ret;
    }
    public Vector forwardPass(Vector previous){
        savedinput = previous;
        Vector values = ((previous.toMatrix()).times(weights)).toVector().plus(bias);
        deltas = sigmoidprime(values);
        return sigmoid(values);
    }
    public Vector sigmoidprime(Vector in){
        double[] ret = new double[in.values.length];
        for(int i=0;i<in.values.length;i++){
            ret[i] = 1/(1+Math.exp(-in.values[i]));
            ret[i] = ret[i]*(1-ret[i]);
        }
        return new Vector(ret);
    }
    public Vector sigmoid(Vector in){
        double[] ret = new double[in.values.length];
        for(int i=0;i<in.values.length;i++){
            ret[i] = 1/(1+Math.exp(-in.values[i]));
        }
        return new Vector(ret);
    }
    
}