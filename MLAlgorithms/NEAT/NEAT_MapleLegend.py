import numpy as np
import cv2


#constants for NEAT
class constants(object):
    POPULATION = 300    

    DELTADISJOINT = 2.0
    DELTAWEIGHTS = .4
    DELTATHRESHOLD = 1.0

    STALESPECIES = 15

    MUTATECONNECTIONCHANCE = .25
    PERTURBCHANCE = 0.9
    CROSSOVERCHANCE = .75
    LINKMUTATIONCHANCE = 2.0
    NODEMUTATIONCHANCE = 0.5
    BIASMUTATIONCHANCE = .4
    STEPSIZE = .1
    
    DISABLEMUTATIONCHANCE = 0.4
    ENABLEMUTATIONCHANCE = .2

    TIMEOUTCONSTANT = 20

    MAXNODES = 100000

#model params (genome)
class Individual(object):
    def __init__(self):
        self.geneLength = 64
        self.genes = np.round(np.random.random(self.geneLength))
        #Cache
        self.fitness = 0

    def __str__(self):
        geneString = ''.join(str(int(x)) for x in self.genes)
        return geneString
    
    def getFitness(self):
        if(self.fitness == 0):
            self.fitness = getFitness(self)
        return self.fitness

    #randomize 
    def generateIndividual(self):
        self.genes = np.random.bytes(self.geneLength)

    def getGene(self,location):
        return genes[location]
    
    def setGene(self,location, newVal):
        self.genes[location] = newVal
        fitness = 0

    def getGeneLength(self):
        return self.geneLength
    
    def setGeneLength(self, length):
        self.geneLength = length

#fitness function


#
