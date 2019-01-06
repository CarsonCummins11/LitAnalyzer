import java.io.Serializable;

public class GRUCell implements Serializable{
    private static final long serialVersionUID = 1L;
    Matrix dweight;
    Matrix eweight;
    Matrix fweight;
    Matrix qweight;
    Matrix aweight;
    Matrix bweight;
    Matrix zweight;
    Vector activatedValue;
    Vector previous_input;
    Vector delta_d;
    Vector delta_e;
    Vector delta_f;
    Vector delta_q;
    Vector delta_a;
    Vector delta_b;
    Vector delta_z;
    Vector delta_p;
	Matrix dweight_fut=null;
	Matrix eweight_fut=null;
	Matrix fweight_fut=null;
	Matrix qweight_fut=null;
	Matrix aweight_fut=null;
	Matrix bweight_fut=null;
	Matrix zweight_fut=null;
    public GRUCell(int inputsize){
        dweight = Matrix.random(inputsize,inputsize);
        eweight = Matrix.random(inputsize,inputsize);
        fweight = Matrix.random(inputsize,inputsize);
        qweight = Matrix.random(inputsize,inputsize);
        aweight = Matrix.random(inputsize,inputsize);
        bweight = Matrix.random(inputsize,inputsize);
        zweight = Matrix.random(inputsize,inputsize);

    }
    /*
    https://towardsdatascience.com/understanding-gru-networks-2ef37df6c9be
    Follows the diagram on here
    */
    public Vector activate(Vector p,Vector x){
        //multiply in weights, we're not gonna do biases
        Vector d = dweight.weightmultipy(p);
        Vector e = eweight.weightmultipy(p);
        Vector f = fweight.weightmultipy(p);
        Vector q = qweight.weightmultipy(p);
        Vector a = aweight.weightmultipy(x);
        Vector b = bweight.weightmultipy(x);
        Vector z = zweight.weightmultipy(x);
        //traverse the GRU cell
        Vector c = z.plus(d);
        Vector L = sigmoid(c);
        Vector g = b.plus(e);
        Vector h = sigmoid(g);
        Vector i = f.hadamardProduct(h);
        Vector j = a.plus(i);
        Vector k = tanh(j);
        Vector m = L.hadamardProduct(k);
        Vector r = L.hadamardProduct(q);
        //return the result
        return m.plus(r);
    }
    public Vector activate_deltas(Vector p,Vector x){
    	previous_input = p;
        activatedValue = activate(p, x);
        delta_d = activateprime_d(p, x);
        delta_e = activateprime_e(p, x);
        delta_f = activateprime_f(p, x);
        delta_q = activateprime_q(p, x);
        delta_a = activateprime_a(p, x);
        delta_b = activateprime_b(p, x);
        delta_z = activateprime_z(p, x);
        delta_p = activateprime_p(p,x);
        return activatedValue;
    }
    public Vector activateprime_p(Vector p,Vector x){
        Vector d = dweight.weightmultipy(p);
        Vector e = eweight.weightmultipy(p);
        Vector f = fweight.weightmultipy(p);
        Vector a = aweight.weightmultipy(x);
        Vector b = bweight.weightmultipy(x);
        Vector z = zweight.weightmultipy(x);
        
        Vector aa = sigmoidprime(z.plus(d));
        Vector ab = sigmoid(z.plus(d));
        Vector bb = sigmoid(b.plus(e));
        Vector ac = a.plus(f.hadamardProduct(bb));
        Vector ad = tanh(ac);
        Vector ae = tanhPrime(ac);
        Vector ba = sigmoidprime(b.plus(e));

        return aa.hadamardProduct(ad).weightmultipy(dweight).plus(ab.hadamardProduct(ae).hadamardProduct(fweight.weightmultipy(bb).plus(f.hadamardProduct(ba).weightmultipy(eweight))));


    }
    public Vector activateprime_d(Vector p, Vector x){
        //multiply in weights, we're not gonna do biases
        Vector d = dweight.weightmultipy(p);
        Vector e = eweight.weightmultipy(p);
        Vector f = fweight.weightmultipy(p);
        Vector q = qweight.weightmultipy(p);
        Vector a = aweight.weightmultipy(x);
        Vector b = bweight.weightmultipy(x);
        Vector z = zweight.weightmultipy(x);
        Vector g = b.plus(e);
        Vector h = sigmoid(g);
        Vector i = f.hadamardProduct(h);
        Vector mprime = sigmoidprime(z.plus(d)).hadamardProduct(tanh(a.plus(i)));
        Vector rprime = sigmoidprime(z.plus(d)).hadamardProduct(q);
        
        return mprime.plus(rprime);
    }
    //partial derivative for e weights
    public Vector activateprime_e(Vector p,Vector x){
        //multiply in weights, we're not gonna do biases
        Vector d = dweight.weightmultipy(p);
        Vector e = eweight.weightmultipy(p);
        Vector f = fweight.weightmultipy(p);
        Vector a = aweight.weightmultipy(x);
        Vector b = bweight.weightmultipy(x);
        Vector z = zweight.weightmultipy(x);

        Vector c = z.plus(d);
        Vector L = sigmoid(c);
        Vector g = b.plus(e);
        Vector h = sigmoid(g);
        Vector i = f.hadamardProduct(h);
        Vector j = a.plus(i);

        Vector kprime = tanhPrime(j).hadamardProduct(f.hadamardProduct(sigmoidprime(b.plus(e))));
        Vector mprime = L.hadamardProduct(kprime);
        return mprime;
    }
    //partial derivative for f weights
    public Vector activateprime_f(Vector p,Vector x){
        //multiply in weights, we're not gonna do biases
        Vector d = dweight.weightmultipy(p);
        Vector e = eweight.weightmultipy(p);
        Vector f = fweight.weightmultipy(p);
        Vector a = aweight.weightmultipy(x);
        Vector b = bweight.weightmultipy(x);
        Vector z = zweight.weightmultipy(x);

        Vector c = z.plus(d);
        Vector L = sigmoid(c);
        Vector g = b.plus(e);
        Vector h = sigmoid(g);
        Vector i = f.hadamardProduct(h);
        Vector j = a.plus(i);

        Vector mprime = L.hadamardProduct(tanhPrime(j)).hadamardProduct(h);
        return mprime;
    }
    //partial derivative for q weights
    public Vector activateprime_q(Vector p,Vector x){
        //multiply in weights, we're not gonna do biases
        Vector d = dweight.weightmultipy(p);
        Vector z = zweight.weightmultipy(x);

        Vector c = z.plus(d);
        Vector L = sigmoid(c);

        return L;
    }
    //partial derivative for a weights
    public Vector activateprime_a(Vector p,Vector x){
        //multiply in weights, we're not gonna do biases
        Vector d = dweight.weightmultipy(p);
        Vector e = eweight.weightmultipy(p);
        Vector f = fweight.weightmultipy(p);
        Vector a = aweight.weightmultipy(x);
        Vector b = bweight.weightmultipy(x);
        Vector z = zweight.weightmultipy(x);

        Vector c = z.plus(d);
        Vector L = sigmoid(c);
        Vector g = b.plus(e);
        Vector h = sigmoid(g);
        Vector i = f.hadamardProduct(h);
        Vector j = a.plus(i);
        
        Vector mprime = L.hadamardProduct(tanhPrime(j));
        return mprime;
    }
    //partial derivative for b weights
    public Vector activateprime_b(Vector p,Vector x){
        //multiply in weights, we're not gonna do biases
        Vector d = dweight.weightmultipy(p);
        Vector e = eweight.weightmultipy(p);
        Vector f = fweight.weightmultipy(p);
        Vector a = aweight.weightmultipy(x);
        Vector b = bweight.weightmultipy(x);
        Vector z = zweight.weightmultipy(x);

        Vector c = z.plus(d);
        Vector L = sigmoid(c);
        Vector g = b.plus(e);
        Vector h = sigmoid(g);
        Vector i = f.hadamardProduct(h);
        Vector j = a.plus(i);

        Vector kprime = tanhPrime(j).hadamardProduct(f).hadamardProduct(sigmoidprime(b.plus(e)));
        Vector mprime = L.hadamardProduct(kprime);
        return mprime;
    }
    //partial derivative for z weights
    public Vector activateprime_z(Vector p,Vector x){
        //these should be the same because they're just added together at the beginning
        return activateprime_d(p, x);
    }
    public Vector tanhPrime(Vector in){
        double[] ret = new double[in.values.length];
        for(int i=0;i<in.values.length;i++){
            ret[i] = Math.tanh(in.values[i]);
            ret[i] = 1-(ret[i]*ret[i]);
        }
        return new Vector(ret);
    }
    public Vector sigmoidprime(Vector in){
        double[] ret = new double[in.values.length];
        for(int i=0;i<in.values.length;i++){
            ret[i] = 1/(1+Math.exp(-in.values[i]));
            ret[i] = ret[i]*(1-ret[i]);
        }
        return new Vector(ret);
    }
    public Vector tanh(Vector in){
        double[] ret = new double[in.values.length];
        for(int i=0;i<in.values.length;i++){
            ret[i] = Math.tanh(in.values[i]);
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