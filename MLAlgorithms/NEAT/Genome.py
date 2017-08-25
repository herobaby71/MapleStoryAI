import numpy as np
import copy
import random
import operator
from NEAT_MapleLegend import constants
from Pool import Pool



pool = Pool()
def newInnovation():
    pool.innovation = pool.innovation+1
    return pool.innovation

class Genome(object):
    class Neuron(object):
        def __init__(self):
            self.incoming = {}
            self.value = 0.0
    class Gene(object):
        def __init__(self, into = 0, out = 0, weight = 0.0, enable = True, innovation = 0):
            self.into = into
            self.out = out
            self.weight = weight
            self.enable = enable
            self.innovation = innovation

        def copy(self):
            return copy.deepcopy(self)
    def __init__(self, Inputs = 0, Outputs = 0):
        self.Inputs = Inputs
        self.Outputs = Outputs
        self.genes = []
        self.fitness = 0
        self.adjusted_fitness = 0
        self.network = {}
        self.max_neuron = 0
        self.global_rank = 0
        self.mutation_rates = self.initializeMutationRates()
    def initializeMutationRates(self):
        mut_rates = {}
        mut_rates["connections"] = constants.MUTATECONNECTIONCHANCE
        mut_rates["link"] = constants.LINKMUTATIONCHANCE
        mut_rates["bias"] = constants.BIASMUTATIONCHANCE
        mut_rates["node"] = constants.NODEMUTATIONCHANCE
        mut_rates["enable"] = constants.ENABLEMUTATIONCHANCE
        mut_rates["disable"] = constants.DISABLEMUTATIONCHANCE
        mut_rates["step"] = constants.STEPSIZE
        return mut_rates

    def generateNetwork(self):
        """
            create a network of neurons based on the genes given
        """
        network = self.network
        genes = self.genes
        Inputs = self.Inputs
        Outputs = self.Outputs
        
        network["neurons"] = {}

        for i in range(Inputs):
            network["neurons"][i] = Neuron()

        #Outputs neuron start after MaxNodes
        for o in range(Outputs):
            network["neurons"][constants.MAXNODES+o] = Neuron()


        #sort the genes by their output values
        genes = sorted(genes.items()), key = lambda x: x.out, reverse = True)

        for i,gene in enumerate(genes):
            if(gene.enable):
                #create the neuron if not exists
                if(network["neurons"].get(gene.out) is None):
                    network["neurons"][gene.out] = Neuron()

                neuron = network["neurons"][gene.out]
                neuron.incoming.append(gene)

                if(network["neurons"].get(gene.into) is None):
                    network["neurons"][gene.into] = Neuron()

    #evaluate the neuronetwork with given inputs
    def evaluateNetwork(self, inputs):
        network = self.network
        maxNodes = constants.MAXNODES
        
        #check if the inputs is valid
        if(not(len(inputs) == self.Inputs)):
            print("input is invalid, no outputs")
            return

        #set the neuronetwork parameters
        for i in range(len(inputs)):
            network["neurons"][i].value = inputs[i]

        for key, neuron in network["neurons"].item():
            total = 0
            for j in len(neuron.incoming):
                incoming_gene = neuron.incoming[j]
                other = network["neurons"][incoming_gene.into]
                total = total + incoming_gene.weight * other.value

            if (len(neuron.incoming) > 0):
                neuron.value = sigmoid(total)

        #evaluate the outputs = {}
        outputs = []* self.Outputs
        for o in range(self.Outputs):
            if(network["neurons"][maxNodes+o].value > 0):
                outputs[o] = True
            else:
                outputs[o] = False

        return outputs
    
    def containsLink(self, link):
        """
           A linear check to see if the neuronet has the link
           link: a dictionary with "into": node and "out": node
        """
        genes = len(genes)
        for gene in range(len(genes)):
            if(gene.into == link["into"] and gene.out == link["out"]):
                return True
        return False
    
    def copy(self):
        return copy.deepcopy(self)

    #mutations
    def crossover(g1,g2):
        #always prioritize the genome with the most fitness
        if(g1.fitness < g2.fitness):
            g1, g2 = g2, g1

        #create the new child crossover genome
        crossoverGen = Genome()

        innovations2 = {}
        for i in range(len(g2.genes)):
            gene = g2.genes[i]
            innovations2[gene.innovation] = gene            

    def getRandomNeuron(self, fromInput = True):
        """
           Fet a random neuron from the neuronetwork
           fromInput specifies whether the neuron is going to be from the input layer or not
        """
        neurons = {}
        #check if we want to take input layer into account when randomly pick
        if(fromInput):
            for i in range(self.Inputs):
                neurons[i] = True
        #add the rest of the neurons into the dictionary
        for o in range(self.Outputs):
            neurons[constants.MAXNODES+o] = True

        #if fromInput is True, take all neurons in the connections
        #otherwise, only take neurons that is not from the input layer
        for i in range(len(self.genes)):
            if(fromInput or self.genes[i].into > self.Inputs):
                neurons[genes[i].into] = True
            if(fromInput or self.genes[i].out > self.Inputs):
                neurons[genes[i].out] = True

        rand = round(random.random()*(len(neurons)-1))+1
        for nodeNum, neuron in neurons.item():
            rand-=1
            if(rand == 0):
                return nodeNum

        return 0
    #link Mutation
    def mutateLink(self, forceBias):
        """
            Mutate the link or gene by adding connection between two neurons
            forceBias: if True, then the into connection will be from the bias (node #Inputs)
            Since this mutation add a connection, add 1 to innovation
            
        """
        #neuron1 is randomly picked from all nodes
        #neuron2 is randomly picked from all nodes except the one from the input layer
        neuron1 = self.getRandomNeuron(True)
        neuron2 = self.getRandomNeuron(False)
        Inputs = self.Inputs
        #if the neurons selected are in the same input layer than do nothing
        if(neuron1 <= Inputs and neuron2 <= Inputs):
            return
        #if neuron2 is the input
        #we want it to be named neuron1 since it will be the input
        if(neuron2 <= Inputs):
            neuron1,neuron2 = neuron2, neuron1

        newLink = Gene(neuron1, neuron2)

        if(forceBias):
            newLink.into = Inputs-1

        #check if the link is already available
        if(containsLink(self.genes, newLink)):
            return

        newLink.innovation = newInnovation()
        newLink.weight = random.random()*4-2

        #add the new link to genes array
        self.genes.append(newLink)

    def mutateNode(self):
        """
           Add a neuron to the genome based on a current 
        """
        #if the genes array is empty.
        if(len(self.genes) ==0 ):
            return

        #add a node
        self.max_neuron = self.max_neuron+1

        gene = self.genes[round(random.random()* (len(genome.genes)-1))+ 1)]              
        #check to see if genemutation is enable
        if(not(gene.enable)): return
        #prevent furthur mutation in the next iteration because we will copy from it
        gene.enable = False

        #create new genes based on the picked one
        #add connections (link) between itself and the new neurons
        gene_temp1 = gene.copy()
        gene_temp1.out = self.maxneuron
        gene_temp1.weight = 1
        gene_temp1.innovation = newInnovation()
        gene_temp1.enable = True
        self.genes.append(gene_temp1)

        #gene_temp2 maintains the weight originality
        gene_temp2 = gene.copy()
        gene_temp2.into = self.max_neuron
        gene_temp2.innovation = newInovation()
        gene_temp2.enable = True
        self.genes.append(gene_temp2)

    def enableDisableMutate(self, enable):
        candidates = []
        for gene in self.genes:
            if(gene.enable == not enable):
                candidates.append(gene)
        if(len(candidates) == 0):
            return
        gene = candidates[round(random.random()* len(genome.genes))]
        gene.enable = not gene.enable
        
    def mutate(self):

        #go through all mutations and decrease or decrease the rate by 50/50 chance
        for mutation, prob in self.mutation_rates.items():
            self.mutation_rates[mutation] = .95 * prob if random.random() < .5 else self.mutation_rates[mutation] = 1.05263 * prob

        #mutate connections in the genes by connectionmutationchance
        
                
        #mutate links
