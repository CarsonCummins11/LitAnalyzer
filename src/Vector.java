
import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Vector implements Serializable{
    private static final long serialVersionUID = 1L;
    double[] values;
public Vector(int size){
    values = new double[size];
}
public Vector(double[] d){
    values = d;
}
public double[] getValues(){
    return values;
}
public Vector weightmultipy(Matrix m){
    return (toMatrix().times(m)).toVector();
}
public void setValue(int index, double value){
    values[index] = value;
}
public static Vector random(int size){
    Vector ret = new Vector(size);
    for(int i =0; i<size; i++){
        ret.setValue(i,Math.random());
    }
    return ret;
}
public double dotProduct(Vector v){
    double[] ovals = v.getValues();
    double sum = 0;
    for (int i=0; i<values.length ;i++){
        sum+=ovals[i]*values[i];
    }
    return sum;
}
public Matrix toMatrix(){
    Matrix ret = new Matrix(1,values.length);
    for(int i=0; i<values.length; i++){
        ret.data[0][i] = values[i];
    }
    return ret;
}
public Matrix toMatrixVertical(){
    Matrix ret = new Matrix(values.length,1);
    for(int i=0; i<values.length; i++){
        ret.data[i][0] = values[i];
    }
    return ret;
}
public Vector scale(double scalar){
    Vector ret= new Vector(this.getValues().clone());
    for (int i = 0; i<ret.getValues().length; i++){
        ret.setValue(i, ret.getValues()[i]*scalar);
    }   
    return ret;
}
public Vector plus(Vector v){
return toMatrix().plus(v.toMatrix()).toVector();
}
public Vector hadamardProduct(Vector v){
    Vector ret = new Vector(v.values.length);
    for(int i=0; i<values.length; i++){
        ret.values[i] = v.values[i]*values[i];
    }
    return ret;
}
public double sum(){
    double sum = 0;
    for(double d: values){
        sum+=d;
    }
    return sum;
}
public void print(){
    String tp = "";
    for (double d:values){
        tp+=d+",";
    }
    System.out.println(tp);
}
public INDArray toINDArray() {
	return Nd4j.create(values);
}
public Matrix toMatrixByDivision(Vector vector) {
	// TODO Auto-generated method stub
	return null;
}
}