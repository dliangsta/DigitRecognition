///////////////////////////////////////////////////////////////////////////////
//  
// Main Class File:  NNBuilder.java
// File:             Instance.java
// Semester:         CS540 Artificial Intelligence Summer 2016
// Author:           David Liang dliang23@wisc.edu
//
//////////////////////////////////////////////////////////////////////////////

import java.util.*;

public class NeuralNetwork
{
	public ArrayList<Node> inputNodes = null;//list of the output layer nodes.
	public ArrayList<Node> hiddenNodes = null;//list of the hidden layer nodes
	public ArrayList<Node> outputNodes = null;// list of the output layer nodes
	public ArrayList<Instance> trainingSet = null;//the training set

	Double learningRate = 1.0; // variable to store the learning rate
	int maxEpoch = 1; // variable to store the maximum number of epochs

	private double[] errorK, deltaK;
	private double[][] deltaJK, deltaIJ, weightsJK, weightsIJ;
	private boolean crossValidate = false;
	private int kFolds = 5;
	/**
	 * This constructor creates the nodes necessary for the neural network and connects the nodes of different layers.
	 * After calling the constructor the last node of both inputNodes and  hiddenNodes will be bias nodes. 
	 */
	public NeuralNetwork(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Double [][]hiddenWeights, Double[][] outputWeights) 
	{
		this.trainingSet = trainingSet;
		this.learningRate = learningRate;
		this.maxEpoch = maxEpoch;
		//input layer nodes
		inputNodes = new ArrayList<Node>();
		int inputNodeCount =trainingSet.get(0).attributes.size();
		int outputNodeCount = trainingSet.get(0).classValues.size();
		for(int i = 0; i < inputNodeCount; i++) 
		{
			Node node = new Node(0);
			inputNodes.add(node);
		}
		// bias node from input layer to hidden
		Node biasToHidden = new Node(1);
		inputNodes.add(biasToHidden);
		// hidden layer nodes
		hiddenNodes = new ArrayList<Node>();
		for (int i = 0; i < hiddenNodeCount; i++) 
		{
			Node node = new Node(2);
			// Connecting hidden layer nodes with input layer nodes
			for(int j = 0; j < inputNodes.size(); j++) 
			{
				NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
				node.parents.add(nwp);
			}
			hiddenNodes.add(node);
		}
		// bias node from hidden layer to output
		Node biasToOutput = new Node(3);
		hiddenNodes.add(biasToOutput);
		// Output node layer
		outputNodes = new ArrayList<Node>();
		for(int i = 0; i < outputNodeCount; i++) 
		{
			Node node = new Node(4);
			// Connecting output layer nodes with hidden layer nodes
			for(int j = 0; j < hiddenNodes.size(); j++)
			{
				NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
				node.parents.add(nwp);
			}	
			outputNodes.add(node);
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
		// find max output
		for (int i = 0; i < outputNodes.size(); i++)
		{
			if ((int)(outputNodes.get(i).getOutput()*10) >= maxOutput) 
			{
				maxOutput = (int)(outputNodes.get(i).getOutput()*10);
				outputIndex = i;
			}
		}

		return outputIndex;
	}

	/**
	 * Train the neural networks with the given parameters
	 */
	public void train()
	{
		if (!crossValidate)
		{
			int iteration = 0;
			while (iteration++ < maxEpoch )
			{
				for(int h = 0; h < trainingSet.size(); h++) // for each instance, update weights
				{
					updateWeights(trainingSet.get(h)); // calculate weights, deltas, errors and save in instance variables and then update variables
				}
			}
		}
		else
		{
			crossValidate();
		}
	}

	/**
	 * Calculate errors, collect weights, and calculate deltas.
	 */
	private void updateWeights(Instance inst)
	{
		calculateErrors(inst);
		collectWeights();
		calculateDeltas();
		applyChanges();
	}

	/**
	 * Calculate the error of each output node.
	 */
	private void calculateErrors(Instance inst) 
	{
		calculateOutputForInstance(inst); // calculate output node outputs
		errorK = new double[outputNodes.size()];
		for (int k = 0; k < outputNodes.size(); k++) // calculate error for each output node
		{
			errorK[k] = inst.classValues.get(k) - outputNodes.get(k).getOutput();
		}
	}

	/**
	 * Collect weights from each arc.
	 */
	private void collectWeights()
	{
		weightsJK = new double[hiddenNodes.size()][outputNodes.size()];
		weightsIJ = new double[inputNodes.size()][hiddenNodes.size() - 1];
		for (int k = 0; k < outputNodes.size(); k++)  // each output node
		{
			for (int j = 0; j < outputNodes.get(k).parents.size(); j++) // each parent of output i.e. each hidden node 
			{
				weightsJK[j][k] = outputNodes.get(k).parents.get(j).weight;
			}
		}
		for (int j = 0; j < hiddenNodes.size() - 1; j++) // for each hidden node 
		{
			for (int i = 0; i < hiddenNodes.get(j).parents.size(); i++)  // each parent of hidden i.e. each input node
			{
				weightsIJ[i][j] = hiddenNodes.get(j).parents.get(i).weight;
			}
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
	private void applyChanges() 
	{
		for (int k = 0; k < outputNodes.size(); k++)
		{
			for(int j = 0; j < outputNodes.get(k).parents.size(); j++) 
			{
				outputNodes.get(k).parents.get(j).weight += deltaJK[j][k];
			}
		}
		for (int j = 0; j < hiddenNodes.size() - 1; j++)
		{
			for (int i = 0; i < hiddenNodes.get(j).parents.size(); i++)
			{
				hiddenNodes.get(j).parents.get(i).weight += deltaIJ[i][j];
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

		for(int j = 0; j < instances.size(); j++) 
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

	private void crossValidate()
	{
		double[] accuracies = new double[kFolds];
		ArrayList<ArrayList<Instance>> instancesList = partitionDataSet(trainingSet, kFolds);
		
		for (int z = 0; z < instancesList.size(); z++)
		{
			Random r = new Random();

			for (int k = 0; k < outputNodes.size(); k++) // set weights to new randomized weights
			{
				for(int j = 0; j < outputNodes.get(k).parents.size(); j++) 
				{
					outputNodes.get(k).parents.get(j).weight = r.nextDouble()*0.01;;
				}
			}
			for (int j = 0; j < hiddenNodes.size() - 1; j++)
			{
				for (int i = 0; i < hiddenNodes.get(j).parents.size(); i++)
				{
					hiddenNodes.get(j).parents.get(i).weight = r.nextDouble()*0.01;
				}
			}

			int iteration = 0;
			while (iteration++ < maxEpoch ) // train
			{
				for(int h = 0; h < instancesList.get(z).size(); h++) // for each instance, update weights
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

			System.out.println("Fold " + (z+1) + " accuracy: " + count / instancesList.get(z).size());
		}

		double average = 0;

		for(int i = 0; i < accuracies.length; i++) // add up accuracies
		{
			average += accuracies[i];
		}
		
		average = average / kFolds; // find average
		
		System.out.println(kFolds + "-fold cross validation average accuracy: " + average);
	}
}
