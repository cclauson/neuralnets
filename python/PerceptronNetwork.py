import random
import math

def roundedList(myList):
    return [round(elem, 4) for elem in myList]

#A perceptron maintains a list, which should always
#be a list of references to other perceptrons
class Perceptron:
    """
    A Perceptron is a single node in a DAG
    used to represent the topology of a
    PerceptronNetwork
    """

    def __init__(self, name):
        """
        Creates a new Perceptron with the given name

        :param name: Name of this Perceptron node
        :type name: string
        """
        self.edges = []
        self.name = name

    def addEdge(self, edgeDest):
        """
        Adds a new edge from this Perceptron to the
        given Perceptron

        :param edgeDest: Destination of edge, source is this/self
        :type edgeDest: Perceptron
        """
        self.edges.append(edgeDest)

    def __str__(self):
        return self.name

class PerceptronNetwork:
    """
    A perceptron network represents a topology
    of perceptrons, with designated inputs and
    designated outputs.  It is an immutable object.

    The weights are not stored as state in the
    network, but rather are passed as parameters
    to operations such as training and classification.

    Also note that input perceptrons to not actually
    act as true perceptrons, their "output value" is
    in fact just the set of values assigned by the
    input vector
    """

    #utility function that DFS's to traverse the
    #entire DAG, the edgeProcessor object is called
    #once for each edge.  Also, this function checks
    #graph topology restrictions stated in the doc
    #for the constructor.
    def _iterateEdges(self, edgeProcessor):
        #In order to populate this data structure,
        #we DFS from each input node, we also
        #check for cycles, there shouldn't be any

        #we will never modify these, this
        #is just so we can quickly check
        inputSet = set(self.inputList)
        outputSet = set(self.outputList)

        visited = set()
        for inputNode in self.inputList:
            stack = []
            inStack = set()
            stack.append(inputNode)
            inStack.add(inputNode)
            while len(stack) > 0:
                node = stack.pop()
                #wait until after iterating over
                #children to remove from inStack

                #one could theoretically imagine a perceptron network
                #where an output is also fed into another node to
                #eventually produce another output, I've never seen
                #this, so let's flag it as an error
                if node in outputSet and len(node.edges) > 0:
                        raise ValueError("Found edge beginning at output "
                                         "node, this is not supported")
                for childNode in node.edges:
                    if childNode in inputSet:
                        raise ValueError("Found edge terminating at input "
                                         "node, this is not supported")
                    if childNode in inStack:
                        raise ValueError("Given network contains a cycle")
                    #this is where we actually process the edge
                    edgeProcessor(node, childNode)
                    if not childNode in visited:
                        inStack.add(childNode)
                        stack.append(childNode)
                inStack.remove(node)
                visited.add(node)

        if not inputSet.issubset(visited):
            #this should be true for every argument
            #unless our code has an error
            raise RuntimeError("Sanity check failure: not every input "
                               "node visited")
        if not outputSet.issubset(visited):
            #this can happen if we were passed a graph
            #with topology such that some output node
            #is not reachable from any input node, flag
            #as error because we cannot compute a value
            #for that output
            raise ValueError("Found unreachable output node, "
                             "which is an error")

    #common logic for traversing forward or backward
    def _topoTraverse(self, initialSet, predecessorLookup,
                      successorLookup, nodeProcessor):
        visitedSet = set()
        activeSet = set(initialSet)
        while len(activeSet) > 0:
            n = activeSet.pop()
            predecessors = predecessorLookup(n)
            nodeProcessor(n, predecessors)
            visitedSet.add(n)
            for childNode in successorLookup(n):
                if childNode in visitedSet or childNode in activeSet:
                    continue
                boolAllPredsVisited = True
                for pred in predecessorLookup(childNode):
                    if not pred in visitedSet:
                        boolAllPredsVisited = False
                if boolAllPredsVisited:
                    activeSet.add(childNode)
        

    def _topoTraverseForward(self, nodeProcessor):
        def lookupPredecessors(node):
            return self.reverseEdgeMap[node]
        def lookupSuccessors(node):
            return node.edges
        self._topoTraverse(self.inputList, lookupPredecessors,
                           lookupSuccessors, nodeProcessor)

    def _topoTraverseBackward(self, nodeProcessor):
        def lookupPredecessors(node):
            return self.reverseEdgeMap[node]
        def lookupSuccessors(node):
            return node.edges
        self._topoTraverse(self.outputList, lookupSuccessors,
                           lookupPredecessors, nodeProcessor)


    def __init__(self, inputList, outputList):
        """
        Construct a new PerceptronNetwork object with
        the given inputList and outputList.

        The structure should be that of a directed
        acyclic graph (DAG).  There should be no
        edges that terminate at any input, or edges
        that begin at any output.  Additionally,
        every output should be reachable from at
        least one input.

        Note that the size of the input list determines
        the dimensionality of the input vector to the
        net, the size of the output list determines the
        dimensionality of the output vector to the net,
        and the dimensionality of the parameter list is
        equal to the number of edges in the DAG.

        :param inputList: A list of input Perceptrons
        :type inputList: A list of Perceptron objects
        :param outputList: A list of output Perceptrons
        :type outputList: A list of Perceptron objects
        """
        
        self.inputList = inputList
        self.outputList = outputList

        #also let's populate a map that gives
        #the set of nodes that have an edge to
        #a given node
        self.reverseEdgeMap = {}

        def initializeWithEdge(beginNode, endNode):
            if not endNode in self.reverseEdgeMap:
                self.reverseEdgeMap[endNode] = []
            self.reverseEdgeMap[endNode].append(beginNode)
        self._iterateEdges(initializeWithEdge)
        for node in self.inputList:
            self.reverseEdgeMap[node] = []

        #each node will have a constant weight
        self.nodeToWeight = {}
        #each edge will have a weight
        self.edgeToWeight = {}
        #also index nodes as well
        self.nodeIndices = {}
        #number of nodes
        self.nodeNum = 0
        #number of weights
        self.numWeights = 0

        #assign numbers first to all input nodes
        for inputNode in self.inputList:
            self.nodeIndices[inputNode] = self.nodeNum
            self.nodeNum += 1

        inputSet = set(self.inputList)        
        outputSet = set(self.outputList)

        def processNodeAndPredecessors(node, predecessors):
            if not node in inputSet:
                self.nodeToWeight[node] = self.numWeights
                self.numWeights += 1
                if not node in outputSet:
                    self.nodeIndices[node] = self.nodeNum
                    self.nodeNum += 2
            for predNode in predecessors:
                if not predNode in self.edgeToWeight:
                    self.edgeToWeight[predNode] = {}
                self.edgeToWeight[predNode][node] = self.numWeights
                self.numWeights += 1
        self._topoTraverseForward(processNodeAndPredecessors)

        #finally assign numbers to output nodes
        for outputNode in self.outputList:
            self.nodeIndices[outputNode] = self.nodeNum
            self.nodeNum += 1


    def inputDim(self):
        return len(self.inputList)

    def weightDim(self):
        return self.numWeights

    def _computeNodeVals(self, inputVals, weights):

        inputSet = set(self.inputList)
        nodeVals = inputVals[:]
        while len(nodeVals) < self.nodeNum:
            nodeVals.append(0.0)
        def forwardCompute(node, predecessors):
            if node in inputSet:
                return
            s = weights[self.nodeToWeight[node]]
            for predNode in predecessors:
                predVal = nodeVals[self.nodeIndices[predNode]]
                weight = weights[self.edgeToWeight[predNode][node]]
                s += predVal * weight
            s_flat = 1/(1 + math.exp(-s))
            nodeVals[self.nodeIndices[node]] = s_flat
        self._topoTraverseForward(forwardCompute)

        return nodeVals

    def weightMap(self, weights):
        if len(weights) != self.numWeights:
            raise ValueError("weights array has wrong dimensions")
                
        inputSet = set(self.inputList)        

        ret = {}

        def printTheWeights(node, predecessors):
            if node in inputSet:
                return
            constWeight = weights[self.nodeToWeight[node]]
            ret[node.name] = constWeight
            for predNode in predecessors:
                weight = weights[self.edgeToWeight[predNode][node]]
                ret[predNode.name + "->" + node.name] = weight
        self._topoTraverseForward(printTheWeights)

        return ret

    def nodeValMap(self, nodeVals):
        ret = {}
        def getTheNodeVals(node, predecessors):
            ret[node.name] = nodeVals[self.nodeIndices[node]]
        self._topoTraverseForward(getTheNodeVals)
        return ret
    

    def classify(self, inputVals, weights):
        if len(inputVals) != len(self.inputList):
            raise ValueError("input vals array has wrong dimensions")
        if len(weights) != self.numWeights:
            raise ValueError("weights array has wrong dimensions")

        nodeVals = self._computeNodeVals(inputVals, weights)
        return nodeVals[-len(self.outputList):]

    
    def train(self, inputVals, outputVals, weights):
        if len(inputVals) != len(self.inputList):
            raise ValueError("input vals array has wrong dimensions")
        if len(outputVals) != len(self.outputList):
            raise ValueError("output vals array has wrong dimensions")
        if len(weights) != self.numWeights:
            raise ValueError("weights array has wrong dimensions")

        nodeVals = self._computeNodeVals(inputVals, weights)
        inputSet = set(self.inputList)

        #for debug, let's print error function
        computedOutputs = nodeVals[-len(self.outputList):]
        error = 0.0
        for i in range(0, len(outputVals)):
            error += (outputVals[i] - computedOutputs[i])**2

        #create appropriately sized array for weight
        #partial derivatives
        weightDeltas = []
        while len(weightDeltas) < len(weights):
            weightDeltas.append(0.0)
        
        outputIndices = {}
        for i in range(0, len(self.outputList)):
            outputIndices[self.outputList[i]] = i
        errorTerms = []
        while len(errorTerms) < self.nodeNum:
            errorTerms.append(0.0)

        def backPropagate(node, successors):
            #"error term" is used in the Mitchell book, but
            #might not be the best term, what we're interested
            #in is the partial derivative of the overall error
            #with respect to the input of the flattening
            #function for this perceptron.  To figure it out,
            #first we compute the partial derivative of the
            #overall error with respect to the *output*
            #of the flattening function, then multiply
            #by the derivative of the output of the flattening
            #function with respect to its input

            s = 0.0
            nodeIndex = self.nodeIndices[node]
            nodeVal = nodeVals[nodeIndex]
            if node in outputIndices:
                #if this node is an output, then its output
                #has a direct contribution to the error
                #function, so we compute and add it in
                index = outputIndices[node]
                s += nodeVal - outputVals[index]
            for succNode in successors:
                #we basically multiply the sensitivity of the
                #error to the input to the next node by the
                #weight along the edge, this gives the sensitivity
                #of the error to the output of this node
                edgeWeightIndex = self.edgeToWeight[node][succNode]
                edgeWeight = weights[edgeWeightIndex]
                successorErrorTerm = errorTerms[self.nodeIndices[succNode]]
                s += edgeWeight * successorErrorTerm
                #we also have to compute the new weight
                weightDeltas[edgeWeightIndex] = nodeVal * successorErrorTerm
            #now we multiply by derivative of flattening function
            thisErrorTerm = s * nodeVal * (1 - nodeVal)
            errorTerms[self.nodeIndices[node]] = thisErrorTerm
            #finally, since error term is computed for this node,
            #let's compute new constant weight for this node, unless
            #it's an input node, in which case there is none
            if not node in inputSet:
                constWeightIndex = self.nodeToWeight[node]                
                weightDeltas[constWeightIndex] = thisErrorTerm
        self._topoTraverseBackward(backPropagate)
        return (weightDeltas, error)

    def __str__(self):
        #just return a listing of all edges
        edgestrings = []
        def printEdgesForNode(node, predecessors):
            for predNode in predecessors:
                edgestrings.append(predNode.name + "->" + node.name)
        self._topoTraverseForward(printEdgesForNode)
        return ', '.join(edgestrings)



if __name__ == "__main__":
    inputList = []
    outputList = []
    #we use a topology taken from Tom M. Mitchell's
    #*Machine Learning*, specifically in the section
    #on the back propagation algorithm
    for i in range(0, 8):
        inputList.append(Perceptron("input" + str(i)))
        outputList.append(Perceptron("output" + str(i)))
    for i in range(0, 3):
        hidden = Perceptron("hidden" + str(i))
        for j in range(0, 8):
            inputList[j].addEdge(hidden)
            hidden.addEdge(outputList[j])
    network = PerceptronNetwork(inputList, outputList)
    
    #construct vector of initial weights
    weights = []
    for i in range(0, network.weightDim()):
        weights.append((random.random() - 0.5)/10)

    numTrainingVectors = 8
    #construct vectors to use for input/output
    bitvecs = []
    for i in range(0, numTrainingVectors):
        curr = []
        for j in range(0, 8):
            if i == j:
                curr.append(1.0)
            else:
                curr.append(0.0)
        bitvecs.append(curr)

    #training
    numTrainingIterations = 15000
    for i in range(0, numTrainingIterations):
        deltas = []
        while len(deltas) < network.numWeights:
            deltas.append(0.0)
        error = 0.0
        for j in range(0, numTrainingVectors):
            inandout = bitvecs[j]
            weightDeltas, errorPart = network.train(inandout, inandout, weights)
            for k in range(0, len(deltas)):
                deltas[k] += weightDeltas[k]
            error += errorPart
        print("Error: " + str(error));
        for k in range(0, len(deltas)):
            weights[k] -= .05 * deltas[k]

    for bitvec in bitvecs:
        print(str(roundedList(network.classify(bitvec, weights))))
    
    
