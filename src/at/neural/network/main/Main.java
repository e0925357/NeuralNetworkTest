package at.neural.network.main;

import at.neural.network.NeuralNetwork;

/**
 * Created by Stefan on 01.08.2016.
 */
public class Main {
    public static void main(String args[]) {
        //Create neural network
        NeuralNetwork network = new NeuralNetwork(null, new int[]{2,3,1});
        //XOR test set
        double inputs[][] = {{1,1},{1,0},{0,1},{0,0}};
        double expectedOutputs[][] = {{0.1},{0.9},{0.9},{0.1}};

        //Learn XOR
        double error = Double.POSITIVE_INFINITY;
        long startTime = System.currentTimeMillis();
        for(int step = 0; error > 0.00001; step++) {
            error = 0;

            //Train
            for(int sample = 0; sample < inputs.length; sample++) {
                network.backPropagationLearningStep(inputs[sample], expectedOutputs[sample], 0.5);
            }

            //Check
            for(int sample = 0; sample < inputs.length; sample++) {
                double output[] = network.forwardPass(inputs[sample]);

                for(int i = 0; i < output.length; i++) {
                    error += Math.pow(expectedOutputs[sample][i] - output[i],2);
                }
            }

            error = Math.sqrt(error);

            if(step % 100000 == 0 || error <= 0.00001) {
                System.out.println("Step=" + step + "\t,Error=" + error);
            }
        }

        System.out.println("Finished learning in " + ((System.currentTimeMillis() - startTime)/1000.0) + "s");

        //Check
        for(int sample = 0; sample < inputs.length; sample++) {
            double output[] = network.forwardPass(inputs[sample]);

            System.out.print("Input [");

            for(int i = 0; i< inputs[sample].length;i++) {
                if(i != 0) {
                    System.out.print(", ");
                }
                System.out.print(inputs[sample][i]);
            }

            System.out.print("] yields [");

            for(int i = 0; i< output.length;i++) {
                if(i != 0) {
                    System.out.print(", ");
                }
                System.out.print(output[i]);
            }

            System.out.println("]");
        }

        System.out.println("Stating generation of [0.9]...");

        error = Double.POSITIVE_INFINITY;
        startTime = System.currentTimeMillis();
        double input[] = {Math.random(), Math.random()};
        double desiredOutput[] = {0.9};
        for(int step = 0; error > 0.00001; step++) {
            error = 0;

            //Generate
            input = network.backPropagationGenerationStep(desiredOutput, input, 0.5);

            //Check
            double output[] = network.forwardPass(input);

            for(int i = 0; i < output.length; i++) {
                error += Math.pow(desiredOutput[i] - output[i],2);
            }

            error = Math.sqrt(error);

            if(step % 100000 == 0 || error <= 0.00001) {
                System.out.println("Step=" + step + "\t,Error=" + error);
            }
        }

        System.out.println("Finished generating in " + ((System.currentTimeMillis() - startTime)/1000.0) + "s");

        System.out.print("Output [");

        for(int i = 0; i< desiredOutput.length;i++) {
            if(i != 0) {
                System.out.print(", ");
            }
            System.out.print(desiredOutput[i]);
        }

        System.out.print("] generated [");

        for(int i = 0; i< input.length;i++) {
            if(i != 0) {
                System.out.print(", ");
            }
            System.out.print(input[i]);
        }

        System.out.println("]");
    }
}
