
///////////////////////////////////////////////////////////////////////////////
// 
// Title:            Handwritten Digit Recognition 
// Files:            Instance.java, NeuralNetworkBuilder.java, NeuralNetwork.java, Node.java,
//                   NodeWeightPair.java
// Semester:         CS540 Artificial Intelligence Summer 2016
// Author:           David Liang dliang23@wisc.edu
//
///////////////////////////////////////////////////////////////////////////////

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

/**
 * This is the class with the main function.
 */

public class Driver
{
   public static final int NUM_HIDDEN = 10;
   public static final int EPOCHS = 1000;
   public static final int FOLDS = 10;
   public static final double LEARNING_RATE = .001;

   public static void main(String[] args)
   {
      // Checking for correct number of arguments
      if (args.length != 2)
      {
         System.out.println("usage: java NeuralNetworkBuilder <trainFile> <testFile>");
         System.exit(-1);
      }

      // Reading the training set
      ArrayList<Instance> train = getData(args[0]);
      ArrayList<Instance> test = getData(args[1]);

      int numInputs = train.get(0).attributes.size();
      int numOutputs = train.get(0).classValues.size();

      NeuralNetwork nn = new NeuralNetwork(numInputs, NUM_HIDDEN, numOutputs);
      nn.train(train, LEARNING_RATE, EPOCHS);

      double accuracy = nn.accuracy(test);

      System.out.println("Total instances: " + test.size());
      System.out.println("Correctly classified: " + accuracy * test.size());
      System.out.println("Accuracy: " + accuracy);

   }

   // Reads a file and gets the list of instances
   private static ArrayList<Instance> getData(String filename)
   {
      ArrayList<Instance> data = new ArrayList<Instance>();
      BufferedReader in;
      int attributeCount = 0;
      int outputCount = 0;

      try
      {
         in = new BufferedReader(new FileReader(filename));
         while (in.ready())
         {
            String line = in.readLine();
            String prefix = line.substring(0, 2);
            if (prefix.equals("//"))
            {
               // intentionally empty
            }
            else if (prefix.equals("##"))
            {
               attributeCount = Integer.parseInt(line.substring(2));
            }
            else if (prefix.equals("**"))
            {
               outputCount = Integer.parseInt(line.substring(2));
            }
            else
            {
               String[] vals = line.split(" ");
               Instance inst = new Instance();

               for (int i = 0; i < attributeCount; i++)
               {
                  inst.attributes.add(Double.parseDouble(vals[i]));
               }
               for (int i = attributeCount; i < vals.length; i++)
               {
                  inst.classValues.add(Integer.parseInt(vals[i]));
               }

               data.add(inst);
            }

         }

         in.close();

         return data;
      }
      catch (Exception e)
      {
         System.out.println("Could not read instances: " + e);
      }

      return null;
   }

}
