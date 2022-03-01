import java.util.List;

public class Driver {

    //train data
    static double [][] X= {
            {0,0,0},
            {1,0,0},
            {0,1,1},
            {1,1,1}
    };
    //Target
    static double [][] Y= {
            {0},{1},{1},{0}
    };

    public static void main(String[] args) {

        NeuralNetwork nn = new NeuralNetwork(3 ,10,1);

        //Array to hold the results
        Double output;

        //Trains the nn on the data
        nn.fit(X, Y, 50000);


        double [][]  input = {
                {0,0,0},  {1,0,0},
                {0,1,1},
                {1,1,1}
        };
        for(double d[]:input)
        {
            output = nn.predict(d);
            if (nn.predict(d) > 0.5){
                System.out.println(1);
            }else {
                System.out.println(0);
            }
        }

    }

}
