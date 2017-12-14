#!/usr/bin/python

import random
import collections
import math
import sys
from util import *

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
        Extract word features for a string x. Words are delimited by
        whitespace characters only.
        @param string x:
        @return dict: feature vector representation of x.
        Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
        """
    sentence = x
    wordCount = collections.defaultdict(int)
    counter = 0
    for word in sentence:
        wordCount[str(counter)] = word
        counter += 1
    return wordCount;

############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
        Given |trainExamples| and |testExamples| (each one is a list of (x,y)
        pairs), a |featureExtractor| to apply to x, and the number of iterations to
        train |numIters|, the step size |eta|, return the weight vector (sparse
        feature vector) learned.
        
        You should implement stochastic gradient descent.
        
        Note: only use the trainExamples for training!
        You should call evaluatePredictor() on both trainExamples and testExamples
        to see how you're doing as you learn after each iteration.
        '''
    weights = collections.defaultdict(int);
    
    # feature => weight
    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    for t in range (numIters):
        trainLoss = 0
        testLoss = 0
        for i in range (len(trainExamples)):
            x, y = trainExamples[i];
            featureX = featureExtractor(x)
            if featureX == None: print featureX
            if y == None: print "yy"
            if weights == None: print "weights"
            print featureX, "feat"
            zValue = dotProduct(weights, featureX)*y
            gradient = collections.defaultdict(int);
            if zValue < 1:
                for featX in featureX:
                    gradient [featX] = -featureX[featX]*y*eta
            else:
                for featX in featureX:
                    gradient [featX] = featureX[featX]*0
            
            increment(weights, -1, gradient)
            trainLoss += max(0, 1.0-dotProduct(weights,featureX)*y);
        print "Train Loss: ", trainLoss
        for j in range (len(testExamples)):
            a, b = testExamples[j];
            featureA = featureExtractor(a)
            testLoss += max(0, 1.0-dotProduct(weights,featureA)*b);
        print "Test Loss::", testLoss



# END_YOUR_CODE
        return weights

############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
        Return a set of examples (phi(x), y) randomly which are classified correctly by
        |weights|.
        '''
    random.seed(42)
    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        
        num = random.randint(1,len(weights))
        arr = random.sample(weights, num)
        phi = collections.defaultdict(int);
        for key in arr:
            phi[key] = random.randint(1,50) ;
        var = dotProduct(phi, weights);
        if var > 0:
            y = 1;
        else: y = -1;
        
        # END_YOUR_CODE
        return (phi, y)
    return [generateExample() for _ in range(numExamples)]

############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
        Return a function that takes a string |x| and returns a sparse feature
        vector consisting of all n-grams of |x| without spaces.
        EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
        You may assume that n >= 1.
        '''
    def extract(x):
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        words = x.split();
        mapCount = collections.defaultdict(int);
        words = "".join(words);
        for i in range (0, len(words)-n+1):
            mapCount[words[i:i+n]] += 1;
        return mapCount
    # END_YOUR_CODE
    return extract

############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
        examples: list of examples, each example is a string-to-double dict representing a sparse vector.
        K: number of desired clusters. Assume that 0 < K <= |examples|.
        maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
        Return: (length K list of cluster centroids,
        list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
        final reconstruction loss)
       '''
    print examples[10:20]
    UiList = []
    oldAssign = []
    assignmentList = []
    for i in range (0, K):
        UiList.append( examples[i] );
    assignmentList = [0]*len(examples)
    for t in range (0, maxIters):
        finalLoss = 0;
        oldAssign = assignmentList[:]
        sumArray = []
        countArray = [0] * K
        arrCenter = [0] * K
        for k in range(K):
            sumArray.append({})
            arrCenter[k] = dotProduct(UiList[k], UiList[k])
        for f in range (0, len(examples)):
            dotX = dotProduct(examples[f],examples[f])
            arrayNum = [];
            for init in range (0, K):
                numToAdd = 2*dotProduct(examples[f],UiList[init]);
                arrayNum.append(dotX + arrCenter[init] - numToAdd);
            num = min(arrayNum)
            assignmentList[f] = arrayNum.index(num);
            increment(sumArray[arrayNum.index(num)], 1, examples[f])
            countArray[arrayNum.index(num)] += 1
            finalLoss += num
        if oldAssign == assignmentList: break;
        newMu = []
        for j in range(0,K):
            newMu.append({})
            if countArray[j] != 0:
                increment(newMu[j], 1.0 / countArray[j], sumArray[j])
        UiList = newMu;
    return UiList, assignmentList, finalLoss

# BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
# END_YOUR_CODE
