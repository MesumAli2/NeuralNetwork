import java.util.Arrays;

public class NeuralNetwork {

    Matrix weights_ih,weights_ho , bias_h,bias_o;
    double  l_rate=0.01;

    public NeuralNetwork(int i,int h,int o) {
        //Weight input to hidden
        //New Matrix with hidden as rows and Inputs as columns
        weights_ih = new Matrix(h,i);
        //Weight hidden to output
        //New Matrix with output as rows and hidden as columns
        weights_ho = new Matrix(o,h);

        bias_h= new Matrix(h,1);
        bias_o= new Matrix(o,1);

    }

    //Takes the input to nerurons dose the operation, (applying sigmoid act func) then gives output
    public Double predict(double[] X)
    {
        Matrix input = Matrix.fromArray(X);

        //Multiplies all the values in the grid with scalar operation
        //Multiplies all the values in the grid with scalar operation
        Matrix hidden = Matrix.multiply(weights_ih, input);
        //Adds all the values in the grid with scalar operation
        
        hidden.add(bias_h);
        hidden.sigmoid();

        Matrix output = Matrix.multiply(weights_ho,hidden);
        output.add(bias_o);
        output.sigmoid();


        return output.toArray().get(0);
    }


    //Trains the Neural network with the given data
    public void fit(double[][]X,double[][]Y,int epochs)
    {

        for(int i=0;i<epochs;i++)
        {
            int  sampleN =  (int)(Math.random() * Y.length);
            this.train(X[sampleN], Y[sampleN]);
        }
    }



    //Trains the NN to what output should be
    public void train(double [] in,double [] tar)
    {
        //Generating the hidden outputs
        Matrix input = Matrix.fromArray(in);
        Matrix hidden = Matrix.multiply(weights_ih, input);
        hidden.add(bias_h);

        //activation function
        hidden.sigmoid();

        //Generating the outputs Output
        Matrix output = Matrix.multiply(weights_ho,hidden);
        output.add(bias_o);
        output.sigmoid();

        //Converts array to matrix object
        Matrix target = Matrix.fromArray(tar);

        //ErrorOutput = Target - Output
        Matrix error = Matrix.subtract(target, output);

        //Calculate Gradient
        Matrix gradient = output.dsigmoid();
        gradient.multiply(error);
        gradient.multiply(l_rate);

        //Calculate Deltas
        Matrix hidden_T = Matrix.transpose(hidden);
        Matrix who_delta =  Matrix.multiply(gradient, hidden_T);

        //Tweaks the weights for hidden layer
        weights_ho.add(who_delta);
        bias_o.add(gradient);

        //Calculate the hidden layers error
        Matrix who_T = Matrix.transpose(weights_ho);
        Matrix hidden_errors = Matrix.multiply(who_T, error);

       // System.out.println("Hidden layers Error array" +hidden_errors.toArray());

        //Calculate hidden gradient
        Matrix h_gradient = hidden.dsigmoid();
        h_gradient.multiply(hidden_errors);
        h_gradient.multiply(l_rate);

        //Calculate input -> hidden deltas
        Matrix i_T = Matrix.transpose(input);
        Matrix wih_delta = Matrix.multiply(h_gradient, i_T);

        //Tweak the hidden layers weight
        weights_ih.add(wih_delta);
        bias_h.add(h_gradient);

    }


}