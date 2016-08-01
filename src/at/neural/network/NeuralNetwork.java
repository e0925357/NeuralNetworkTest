package at.neural.network;

import at.neural.network.exceptions.NullArgumentException;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Holds information about the nodes in a neural network and offers some methods for manipulating them.
 *
 * @author Stefan
 */
public class NeuralNetwork {
    private final List<Double> edgeWeightList = new ArrayList<>();
    private final List<Double> nodeOutputList = new ArrayList<>();
    private final List<Boolean> nodeOutputSetList = new ArrayList<>();


    private final List<Integer> edgeIndexList = new ArrayList<>();
    private final List<Integer> nodeIndexList = new ArrayList<>();
    private final int layerSize[];

    /**
     * @param r the random generator to use. If set to null a new one will be generated.
     * @param layerSize the number of nodes in each layer. The first layer is the input layer and the last one is the
     *                  output layer. There must be at least two layers and a layer has to contain at least one node.
     */
    public NeuralNetwork(Random r, int layerSize[]) {
        if(layerSize == null) throw new NullArgumentException("nodesSize");
        if(layerSize.length < 2) throw new IllegalArgumentException("There must be at least 2 layers!");

        this.layerSize = layerSize;
        if(r == null) {
            r = new Random();
        }

        for(int layer = 0; layer < layerSize.length-1; layer++) {
            if(layerSize[layer] < 1) {
                throw new IllegalArgumentException("There must be at least one node in a layer, but layer " + layer +
                        "has only " + layerSize[layer]);
            }

            nodeIndexList.add(nodeOutputList.size());

            for(int node = 0; node < layerSize[layer]; node++) {
                edgeIndexList.add(edgeWeightList.size());
                nodeOutputList.add(0.0);
                nodeOutputSetList.add(false);

                for(int edge = 0; edge < layerSize[layer + 1]; edge++) {
                    edgeWeightList.add(r.nextDouble()*2 - 1); //Initialize the network with random weights (-1 to 1)
                }
            }
        }

        //Add size index
        edgeIndexList.add(edgeWeightList.size());

        nodeIndexList.add(nodeOutputList.size());

        for(int node = 0; node < layerSize[layerSize.length-1]; node++) {
            edgeIndexList.add(edgeWeightList.size());
            nodeOutputList.add(0.0);
            nodeOutputSetList.add(false);
        }

        //Add size index
        nodeIndexList.add(nodeOutputList.size());
    }

    /**
     * Computes the output of the neural network for the given input. Also saves the output of each neuron.
     * @param inputValues the input values, must be of the size as the number of input neurons.
     * @return the values of the output neurons.
     */
    public double[] forwardPass(double inputValues[]) {
        if(inputValues.length != layerSize[0]) {
            throw new IllegalArgumentException("There are " + layerSize[0] + "input nodes, but there are " +
                    inputValues.length + " inputs!");
        }

        //Set inputs
        for(int node = 0 ; node < nodeIndexList.get(1); node++) {
            nodeOutputList.set(node, inputValues[node]);
            nodeOutputSetList.set(node, true);
        }

        //propagate vales forward
        for(int layer = 1; layer < layerSize.length; layer++) {
            for (int node = nodeIndexList.get(layer); node < nodeIndexList.get(layer+1); node++) {
                //Sum up net input
                double net = computeNetInput(layer, node - nodeIndexList.get(layer));

                //Activation function
                double output = 1.0/(1.0 + Math.exp(-net));

                //Save output
                nodeOutputList.set(node, output);
                nodeOutputSetList.set(node, true);
            }
        }

        double output[] = new double[layerSize[layerSize.length-1]];

        for(int node = 0; node < layerSize[layerSize.length-1]; node++) {
            output[node] = nodeOutputList.get(nodeIndexList.get(layerSize.length-1) + node);
        }

        return output;
    }

    /**
     * Computes the net input for the given node (the sum of all weighted outputs of the last layer).
     * @param layer the index of the layer of the node
     * @param node the local index of the node in that layer
     * @return the net input of the given node.
     */
    public double computeNetInput(int layer, int node) {
        //Sum up net input
        double net = 0.0;

        for(int llNode = nodeIndexList.get(layer-1); llNode < nodeIndexList.get(layer); llNode++) {
            if(!nodeOutputSetList.get(llNode)) {
                throw new IllegalStateException("The output of the last layer has not been computed yet!");
            }

            int edgeIndex = edgeIndexList.get(llNode) + node;
            net += nodeOutputList.get(llNode)*edgeWeightList.get(edgeIndex);
        }

        return net;
    }

    /**
     * Does a full learning cycle (forward and backward pass) using back propagation once.
     * @param inputValues the input vales to use
     * @param expectedOutput the expected output values
     * @param learningFactor the learning factor. A bigger learning factor means bigger steps in the direction of the
     *                       next local minimum.
     */
    public void backPropagationLearningStep(double inputValues[], double expectedOutput[], double learningFactor) {
        double nodeDelta[] = new double[nodeOutputList.size()];

        //1. Forward pass
        forwardPass(inputValues);

        //2. Back propagation

        //2.1 compute deltas
        for(int layer = layerSize.length-1; layer >= 0; --layer) {
            for(int node = nodeIndexList.get(layer); node < nodeIndexList.get(layer+1); node++) {
                double delta;

                if(layer == layerSize.length-1) {
                    //output layer
                    delta = (nodeOutputList.get(node) - expectedOutput[node - nodeIndexList.get(layer)])*
                            nodeOutputList.get(node)*(1-nodeOutputList.get(node));
                } else {
                    //Hidden layer
                    double propagationSum = 0.0;

                    for(int nlNode = nodeIndexList.get(layer+1); nlNode < nodeIndexList.get(layer+2); nlNode++) {
                        int edgeIndex = edgeIndexList.get(node) + nlNode-nodeIndexList.get(layer+1);
                        propagationSum += nodeDelta[nlNode]*edgeWeightList.get(edgeIndex);
                    }

                    delta = propagationSum*nodeOutputList.get(node)*(1-nodeOutputList.get(node));
                }

                nodeDelta[node] = delta;
            }
        }

        //2.2 compute new weights
        for(int layer = layerSize.length-2; layer >= 0; --layer) {
            for(int node = nodeIndexList.get(layer); node < nodeIndexList.get(layer+1); node++) {
                for(int edge = edgeIndexList.get(node); edge < edgeIndexList.get(node+1); edge++) {
                    int targetNode = nodeIndexList.get(layer+1) + edge - edgeIndexList.get(node);
                    double derivative = nodeOutputList.get(node)*nodeDelta[targetNode];
                    double weight = edgeWeightList.get(edge) - learningFactor*derivative;
                    edgeWeightList.set(edge, weight);
                }
            }
        }
    }

    public double[] backPropagationGenerationStep(double[] output, double[] startInput, double generationFactor) {
        double nodeDelta[] = new double[nodeOutputList.size()];

        //1. Forward pass
        forwardPass(startInput);

        //2. Back propagation

        //2.1 compute deltas
        for(int layer = layerSize.length-1; layer >= 0; --layer) {
            for(int node = nodeIndexList.get(layer); node < nodeIndexList.get(layer+1); node++) {
                double delta;

                if(layer == layerSize.length-1) {
                    //output layer
                    delta = (nodeOutputList.get(node) - output[node - nodeIndexList.get(layer)])*
                            nodeOutputList.get(node)*(1-nodeOutputList.get(node));
                } else {
                    //Hidden layer
                    double propagationSum = 0.0;

                    for(int nlNode = nodeIndexList.get(layer+1); nlNode < nodeIndexList.get(layer+2); nlNode++) {
                        int edgeIndex = edgeIndexList.get(node) + nlNode-nodeIndexList.get(layer+1);
                        propagationSum += nodeDelta[nlNode]*edgeWeightList.get(edgeIndex);
                    }

                    delta = propagationSum*nodeOutputList.get(node)*(1-nodeOutputList.get(node));
                }

                nodeDelta[node] = delta;
            }
        }

        //2.2 compute new input
        double input[] = new double[startInput.length];

        for(int node = 0; node < nodeIndexList.get(1); node++) {
            double propagationSum = 0.0;

            for(int nlNode = nodeIndexList.get(1); nlNode < nodeIndexList.get(2); nlNode++) {
                int edgeIndex = edgeIndexList.get(node) + nlNode-nodeIndexList.get(1);
                propagationSum += nodeDelta[nlNode]*edgeWeightList.get(edgeIndex);
            }

            input[node] = startInput[node] - generationFactor*propagationSum;
        }

        return input;
    }
}
