

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;

public class RecurrentNetwork implements Serializable{
    private static final long serialVersionUID = 1L;
    GRUCell[] cells;
    RecurrentNetwork(int size, int inputsize){
        cells = new GRUCell[size];
        for(int i=0;i<size;i++){
            cells[i] = new GRUCell(inputsize);
        }
    }
    public Vector[] forwardPropogate(Vector[] inputs){
        Vector prev_time = new Vector(inputs[0].values.length);
        ArrayList<Vector> ret = new ArrayList<>();
        for(int i =0; i<prev_time.values.length; i++){
            prev_time.values[i] = 0;
        }
        for(int i=0; i<cells.length; i++){
        prev_time = cells[i].activate(prev_time, inputs[i]);
        ret.add(prev_time);
        }
        return ret.toArray(new Vector[cells.length]);
    }
	public static RecurrentNetwork deserialize(String path) throws IOException, ClassNotFoundException {
		 // Reading the object from a file 
         FileInputStream file = new FileInputStream(path); 
         ObjectInputStream in = new ObjectInputStream(file); 
           
         // Method for deserialization of object 
         RecurrentNetwork ret = (RecurrentNetwork)in.readObject(); 
           
         in.close(); 
         file.close(); 
         return ret;
    }
    public void forwardPass_train(Vector[] inputs){
        Vector prev_time = new Vector(inputs[0].values.length);
        for(int i =0; i<prev_time.values.length; i++){
            prev_time.values[i] = 0;
        }
        for(int i=0; i<cells.length; i++){
        prev_time = cells[i].activate_deltas(prev_time, inputs[i]);
        }
    }
    public double scorePrime(Vector[] output, Vector[] input) {
    	Vector[] yhat = forwardPropogate(input);
    	double diffSum = 0;
    	for(int i = 0; i<output.length; i++) {
    		diffSum+=((yhat[i].plus(output[i].scale(-1))).sum())/yhat[i].values.length;
    	}
    	return diffSum/output.length;
    }
	public void train(Vector[][] inputs, Vector[][] outputs) {
    for(int i = 0; i<inputs.length; i++){
       Vector[] input = inputs[i];
       Vector[] output = outputs[i];
       double sp = scorePrime(output,input);
       double count = 0;
       for(int j = 0; j<cells.length; j++){
    	   Vector d_delta = null;
           Vector e_delta = null;
           Vector f_delta = null;
           Vector q_delta = null;
           Vector a_delta = null;
           Vector b_delta = null;
           Vector z_delta = null;
           for(int k = cells.length-1; k>=j; k--){
        	   
               count++;
        	   Vector d_delta_seg = backpass_d(input, j, k).scale(sp);
               Vector e_delta_seg = backpass_e(input, j, k).scale(sp);
               Vector f_delta_seg = backpass_f(input, j, k).scale(sp);
               Vector q_delta_seg = backpass_q(input, j, k).scale(sp);
               Vector a_delta_seg = backpass_a(input, j, k).scale(sp);
               Vector b_delta_seg = backpass_b(input, j, k).scale(sp);
               Vector z_delta_seg = backpass_z(input, j, k).scale(sp);
               d_delta=d_delta==null?d_delta_seg:d_delta.plus(d_delta_seg);
               e_delta=e_delta==null?e_delta_seg:e_delta.plus(e_delta_seg);
               f_delta=f_delta==null?f_delta_seg:f_delta.plus(f_delta_seg);
               q_delta=q_delta==null?q_delta_seg:q_delta.plus(q_delta_seg);
               a_delta=a_delta==null?a_delta_seg:a_delta.plus(a_delta_seg);
               b_delta=b_delta==null?b_delta_seg:b_delta.plus(b_delta_seg);
               z_delta=z_delta==null?z_delta_seg:z_delta.plus(z_delta_seg);
           }
           Matrix d_delta_matr = d_delta.scale(1/count).toMatrixByDivision(cells[j].previous_input);
           Matrix e_delta_matr = e_delta.scale(1/count).toMatrixByDivision(cells[j].previous_input);
           Matrix f_delta_matr = f_delta.scale(1/count).toMatrixByDivision(cells[j].previous_input);
           Matrix q_delta_matr = q_delta.scale(1/count).toMatrixByDivision(cells[j].previous_input);
           Matrix a_delta_matr = a_delta.scale(1/count).toMatrixByDivision(inputs[i][j]);
           Matrix b_delta_matr = b_delta.scale(1/count).toMatrixByDivision(inputs[i][j]);
           Matrix z_delta_matr = z_delta.scale(1/count).toMatrixByDivision(inputs[i][j]);
           cells[i].dweight_fut = (cells[i].dweight_fut==null?d_delta_matr.scale(-1):cells[i].dweight_fut.minus(d_delta_matr)).scale(1/inputs.length);
           cells[i].eweight_fut = (cells[i].eweight_fut==null?e_delta_matr.scale(-1):cells[i].eweight_fut.minus(e_delta_matr)).scale(1/inputs.length);
           cells[i].fweight_fut = (cells[i].fweight_fut==null?f_delta_matr.scale(-1):cells[i].fweight_fut.minus(f_delta_matr)).scale(1/inputs.length);
           cells[i].qweight_fut = (cells[i].qweight_fut==null?q_delta_matr.scale(-1):cells[i].qweight_fut.minus(q_delta_matr)).scale(1/inputs.length);
           cells[i].aweight_fut = (cells[i].aweight_fut==null?a_delta_matr.scale(-1):cells[i].aweight_fut.minus(a_delta_matr)).scale(1/inputs.length);
           cells[i].bweight_fut = (cells[i].bweight_fut==null?b_delta_matr.scale(-1):cells[i].bweight_fut.minus(b_delta_matr)).scale(1/inputs.length);
           cells[i].zweight_fut = (cells[i].zweight_fut==null?z_delta_matr.scale(-1):cells[i].zweight_fut.minus(z_delta_matr)).scale(1/inputs.length);
       }
       
    }
    descend();
    }
	public void descend() {
		for(GRUCell c : cells) {
			c.dweight = c.dweight.minus(c.dweight_fut);
			c.eweight = c.eweight.minus(c.eweight_fut);
			c.fweight = c.fweight.minus(c.fweight_fut);
			c.qweight = c.qweight.minus(c.qweight_fut);
			c.aweight = c.aweight.minus(c.aweight_fut);
			c.bweight = c.bweight.minus(c.bweight_fut);
			c.zweight = c.zweight.minus(c.zweight_fut);
		}
	}
    public Vector backpass_d(Vector[] input, int cell, int from){
        forwardPass_train(input);
        Vector on = null;
        for(int i = from; i>= cell; i--){
            if(i==cell){
                on = on.toMatrix().times(cells[i].delta_d.toMatrix()).toVector();
            }else{
                if(on==null){
                    on = cells[i].delta_p;
                }else{
                    on = on.toMatrix().times(cells[i].delta_p.toMatrix()).toVector();
                } 
            }
        }
        return on;
    } 
    public Vector backpass_e(Vector[] input, int cell,int from){
        forwardPass_train(input);
        Vector on = null;
        for(int i = from; i>= cell; i--){
            if(i==cell){
                on = on.toMatrix().times(cells[i].delta_e.toMatrix()).toVector();
            }else{
                if(on==null){
                    on = cells[i].delta_p;
                }else{
                    on = on.toMatrix().times(cells[i].delta_p.toMatrix()).toVector();
                } 
            }
        }
        return on;
    } 
    public Vector backpass_f(Vector[] input, int cell,int from){
        forwardPass_train(input);
        Vector on = null;
        for(int i = cells.length-1; i>= cell; i--){
            if(i==cell){
                on = on.toMatrix().times(cells[i].delta_f.toMatrix()).toVector();
            }else{
                if(on==null){
                    on = cells[i].delta_p;
                }else{
                    on = on.toMatrix().times(cells[i].delta_p.toMatrix()).toVector();
                } 
            }
        }
        return on;
    } 
    public Vector backpass_q(Vector[] input, int cell,int from){
        forwardPass_train(input);
        Vector on = null;
        for(int i = from; i>= cell; i--){
            if(i==cell){
                on = on.toMatrix().times(cells[i].delta_q.toMatrix()).toVector();
            }else{
                if(on==null){
                    on = cells[i].delta_p;
                }else{
                    on = on.toMatrix().times(cells[i].delta_p.toMatrix()).toVector();
                } 
            }
        }
        return on;
    } 
    public Vector backpass_a(Vector[] input, int cell,int from){
        forwardPass_train(input);
        Vector on = null;
        for(int i = from; i>= cell; i--){
            if(i==cell){
                on = on.toMatrix().times(cells[i].delta_a.toMatrix()).toVector();
            }else{
                if(on==null){
                    on = cells[i].delta_p;
                }else{
                    on = on.toMatrix().times(cells[i].delta_p.toMatrix()).toVector();
                } 
            }
        }
        return on;
    } 
    public Vector backpass_b(Vector[] input, int cell,int from){
        forwardPass_train(input);
        Vector on = null;
        for(int i = from; i>= cell; i--){
            if(i==cell){
                on = on.toMatrix().times(cells[i].delta_b.toMatrix()).toVector();
            }else{
                if(on==null){
                    on = cells[i].delta_p;
                }else{
                    on = on.toMatrix().times(cells[i].delta_p.toMatrix()).toVector();
                } 
            }
        }
        return on;
    } 
    public Vector backpass_z(Vector[] input, int cell,int from){
        forwardPass_train(input);
        Vector on = null;
        for(int i = from; i>= cell; i--){
            if(i==cell){
                on = on.toMatrix().times(cells[i].delta_z.toMatrix()).toVector();
            }else{
                if(on==null){
                    on = cells[i].delta_p;
                }else{
                    on = on.toMatrix().times(cells[i].delta_p.toMatrix()).toVector();
                } 
            }
        }
        return on;
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