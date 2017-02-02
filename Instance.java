///////////////////////////////////////////////////////////////////////////////
//  
// Main Class File:  NNBuilder.java
// File:             Instance.java
// Semester:         CS540 Artificial Intelligence Summer 2016
// Author:           David Liang dliang23@wisc.edu
//
//////////////////////////////////////////////////////////////////////////////

import java.util.*;

/**
 * Holds data for a particular instance. Attributes are represented as an 
 * ArrayList of Doubles. Class labels are represented as an ArrayList of 
 * Integers. For example, a 3-class instance will have classValues as [0 1 0]
 * meaning this instance has class 1.
 */
 

public class Instance
{
	public ArrayList<Double> attributes;
	public ArrayList<Integer> classValues;
	
	public Instance()
	{
		attributes = new ArrayList<Double>();
		classValues =new ArrayList<Integer>();
	}
	
}
