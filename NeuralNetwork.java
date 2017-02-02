
///////////////////////////////////////////////////////////////////////////////
//  
// Main Class File:  NNBuilder.java
// File:             Instance.java
// Semester:         CS540 Artificial Intelligence Summer 2016
// Author:           David Liang dliang23@wisc.edu
//
//////////////////////////////////////////////////////////////////////////////

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class NeuralNetwork
{
   private static final double WEIGHT = .01;
   private static final boolean crossValidate = true;

   private static double learningRate = .001;

   private ArrayList<Node> inputNodes = null;// list of the output layer nodes.
   private ArrayList<Node> hiddenNodes = null;// list of the hidden layer nodes
   private ArrayList<Node> outputNodes = null;// list of the output layer nodes

   private double[] errorK;
   private double[] deltaK;
   private double[][] deltaJK;
   private double[][] deltaIJ;
   private double[][] weightsIJ;
   private double[][] weightsJK;

   /**
    * This constructor creates the nodes necessary for the neural network and connects the nodes of different layers. After calling the constructor the last node of both inputNodes and hiddenNodes will be bias nodes.
    */
   public NeuralNetwork(int numInputs, int numHiddens, int numOutputs)
   {

      // Reading the weights
      weightsIJ = new double[numInputs + 1][numHiddens];
      weightsJK = new double[weightsIJ.length + 1][numOutputs];
      randomizeWeights(weightsIJ, WEIGHT);
      randomizeWeights(weightsJK, WEIGHT);

      // input layer nodes
      inputNodes = new ArrayList<Node>();

      for (int i = 0; i < numInputs; i++)
      {
         inputNodes.add(new Node(0));
      }
      // bias node from input layer to hidden
      inputNodes.add(new Node(1));

      // hidden layer nodes
      hiddenNodes = new ArrayList<Node>();

      for (int j = 0; j < numHiddens; j++)
      {
         hiddenNodes.add(new Node(2));
         // Connecting hidden layer nodes with input layer nodes
         for (int i = 0; i < inputNodes.size(); i++)
         {
            hiddenNodes.get(j).parents.add(new NodeWeightPair(inputNodes.get(i), weightsIJ[i][j]));
         }
      }
      // bias node from hidden layer to output
      hiddenNodes.add(new Node(3));
      // Output node layer
      outputNodes = new ArrayList<Node>();
      for (int k = 0; k < numOutputs; k++)
      {
         outputNodes.add(new Node(4));
         // Connecting output layer nodes with hidden layer nodes
         for (int j = 0; j < hiddenNodes.size(); j++)
         {
            outputNodes.get(k).parents.add(new NodeWeightPair(hiddenNodes.get(j), weightsJK[j][k]));
         }
      }

      errorK = new double[outputNodes.size()];

   }

   // Gets weights randomly
   public static void randomizeWeights(double[][] weights, double weight)
   {
      Random r = new Random();

      for (int i = 0; i < weights.length; i++)
      {
         for (int j = 0; j < weights[i].length; j++)
         {
            weights[i][j] = r.nextDouble() * weight;
         }
      }
   }

   /**
    * Get the output (index of class value) from the neural network for a single instance.
    */
   public int calculateOutputForInstance(Instance inst)
   {
      for (int i = 0; i < inst.attributes.size(); i++)
      {
         inputNodes.get(i).setInput(inst.attributes.get(i));
      }

      for (int i = 0; i < hiddenNodes.size(); i++)
      {
         hiddenNodes.get(i).calculateOutput();
      }

      for (int i = 0; i < outputNodes.size(); i++)
      {
         outputNodes.get(i).calculateOutput();
      }

      double maxOutput = -1;
      int outputIndex = -1; // index of max output
      // find max output rounded to tenths
      for (int i = 0; i < outputNodes.size(); i++)
      {
         if ((int) (outputNodes.get(i).getOutput() * 10) >= maxOutput)
         {
            maxOutput = (int) (outputNodes.get(i).getOutput() * 10);
            outputIndex = i;
         }
      }

      return outputIndex;
   }

   /**
    * Train the neural networks with the given parameters
    */
   public void train(ArrayList<Instance> train, double learningRate, int epochs)
   {
      NeuralNetwork.learningRate = learningRate;

      if (!crossValidate)
      {
         int iteration = 0;
         while (iteration++ < epochs)
         {
            for (int h = 0; h < train.size(); h++) // for each instance, update weights
            {
               updateWeights(train.get(h)); // calculate weights, deltas, errors and save in instance variables and then update variables
            }
         }
      }
      else
      {
         crossValidate(train, epochs);
      }
   }

   public double accuracy(ArrayList<Instance> test)
   {
      // Reading the training set
      Integer[] outputs = new Integer[test.size()];

      int correct = 0;

      for (int i = 0; i < test.size(); i++)
      {
         // Getting output from network
         outputs[i] = calculateOutputForInstance(test.get(i));

         int actual_idx = -1;

         for (int j = 0; j < test.get(i).classValues.size(); j++)
         {
            if (test.get(i).classValues.get(j) > 0.5)
            {
               actual_idx = j;
            }
         }

         if (outputs[i] == actual_idx)
         {
            correct++;
         }
         else
         {
            System.out.println(i + "th instance got a misclassification, expected: " + actual_idx + ". But actual:" + outputs[i]);
         }
      }
      return (double) correct / test.size();
   }

   /**
    * Calculate errors, collect weights, and calculate deltas.
    */
   private void updateWeights(Instance inst)
   {
      calculateErrors(inst);
      calculateDeltas();
      applyDeltas();
   }

   /**
    * Calculate the error of each output node.
    */
   private void calculateErrors(Instance inst)
   {
      calculateOutputForInstance(inst); // calculate output node outputs

      for (int k = 0; k < outputNodes.size(); k++) // calculate error for each output node
      {
         errorK[k] = inst.classValues.get(k) - outputNodes.get(k).getOutput();
      }
   }

   /**
    * Calculate the change of each arc and save into global variables.
    */
   private void calculateDeltas()
   {
      deltaK = new double[outputNodes.size()];
      deltaJK = new double[hiddenNodes.size()][outputNodes.size()];
      deltaIJ = new double[inputNodes.size()][hiddenNodes.size() - 1];

      for (int k = 0; k < outputNodes.size(); k++)
      {
         if (outputNodes.get(k).getSum() <= 0)
         {
            continue;
         }

         deltaK[k] = errorK[k];

         for (int j = 0; j < outputNodes.get(k).parents.size(); j++) // each parent of output i.e. each hidden node
         {
            deltaJK[j][k] = learningRate * outputNodes.get(k).parents.get(j).node.getOutput() * deltaK[k];
         }
      }

      for (int j = 0; j < hiddenNodes.size() - 1; j++)
      {
         if (hiddenNodes.get(j).getSum() <= 0)
         {
            continue;
         }

         double deltaJ = 0;

         for (int k = 0; k < weightsJK[j].length; k++) // calculate deltaJ
         {
            deltaJ += weightsJK[j][k] * deltaK[k];
         }

         for (int i = 0; i < hiddenNodes.get(j).parents.size(); i++) // calculate deltaIJ
         {
            deltaIJ[i][j] = learningRate * hiddenNodes.get(j).parents.get(i).node.getOutput() * deltaJ;
         }
      }

   }

   /**
    * Apply the changes stored in the deltas.
    */
   private void applyDeltas()
   {
      for (int k = 0; k < outputNodes.size(); k++)
      {
         for (int j = 0; j < outputNodes.get(k).parents.size(); j++)
         {
            weightsJK[j][k] += deltaJK[j][k];
            outputNodes.get(k).parents.get(j).weight = weightsJK[j][k];
         }
      }

      for (int j = 0; j < hiddenNodes.size() - 1; j++)
      {
         for (int i = 0; i < hiddenNodes.get(j).parents.size(); i++)
         {
            weightsIJ[i][j] += deltaIJ[i][j];
            hiddenNodes.get(j).parents.get(i).weight = weightsIJ[i][j];
         }
      }

   }

   private ArrayList<ArrayList<Instance>> partitionDataSet(List<Instance> instances, int numLists)
   {
      ArrayList<ArrayList<Instance>> instancesList = new ArrayList<ArrayList<Instance>>(); // list of instance lists.
      for (int i = 0; i < numLists; i++)
      {
         instancesList.add(new ArrayList<Instance>());
      }

      int k = 0;

      for (int j = 0; j < instances.size(); j++)
      {
         // add each instance to 9 of the 10 lists.
         for (int i = 0; i < numLists; i++)
         {
            if (k == i) // the 1 out of the 10 lists to skip
            {
               continue;
            }
            instancesList.get(i).add(instances.get(j));
         }

         if (k < numLists - 1) // increment k or set to 0
         {
            k++;
         }
         else
         {
            k = 0;
         }
      }

      return instancesList;
   }

   private void crossValidate(ArrayList<Instance> trainingSet, int epochs)
   {
      double[] accuracies = new double[Driver.FOLDS];
      ArrayList<ArrayList<Instance>> instancesList = partitionDataSet(trainingSet, Driver.FOLDS);

      for (int z = 0; z < instancesList.size(); z++)
      {
         Random r = new Random();

         for (int k = 0; k < outputNodes.size(); k++) // set weights to new randomized weights
         {
            for (int j = 0; j < outputNodes.get(k).parents.size(); j++)
            {
               outputNodes.get(k).parents.get(j).weight = r.nextDouble() * 0.01;
               ;
            }
         }
         for (int j = 0; j < hiddenNodes.size() - 1; j++)
         {
            for (int i = 0; i < hiddenNodes.get(j).parents.size(); i++)
            {
               hiddenNodes.get(j).parents.get(i).weight = r.nextDouble() * 0.01;
            }
         }

         int iteration = 0;
         while (iteration++ < epochs) // train
         {
            for (int h = 0; h < instancesList.get(z).size(); h++) // for each instance, update weights
            {
               updateWeights(instancesList.get(z).get(h)); // calculate weights, deltas, errors and save in instance variables and then update variables
            }
         }

         double count = 0;

         for (int h = 0; h < instancesList.get(z).size(); h++)
         { // calculate fold's accuracy
            int maxClass = -1;
            double maxClassValue = -1;
            for (int l = 0; l < instancesList.get(z).get(h).classValues.size(); l++)
            {

               if (instancesList.get(z).get(h).classValues.get(l) > maxClassValue)
               { // find example's class
                  maxClass = l;
                  maxClassValue = instancesList.get(z).get(h).classValues.get(l);
               }
            }
            if (maxClass == calculateOutputForInstance(instancesList.get(z).get(h))) // compare output of network to class value
            {
               count++;
            }
         }

         accuracies[z] = count / instancesList.get(z).size(); // save accuracy to be averaged

         System.out.println("Fold " + (z + 1) + " accuracy: " + count / instancesList.get(z).size());
      }

      double average = 0;

      for (int i = 0; i < accuracies.length; i++) // add up accuracies
      {
         average += accuracies[i];
      }

      average = average / Driver.FOLDS; // find average

      System.out.println(Driver.FOLDS + "-fold cross validation average accuracy: " + average);
   }
}
