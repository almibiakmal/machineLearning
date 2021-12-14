import numpy as np
import sys
import math
import pandas as pd
from sklearn import preprocessing

class mknn:
    def __init__(self, neighbors, fiturTraining, labelTraining, fiturTesting, labelTesting):
        self.neighbors = neighbors
        self.fiturTraining = fiturTraining
        self.labelTraining = labelTraining
        self.fiturTesting = fiturTesting
        self.labelTesting = labelTesting
        self.validitasData = []

        #Normalisasi data training
        normalFiturTraining = preprocessing.normalize(self.fiturTraining, norm='l1')

        jumlahDataTraining = len(normalFiturTraining)
        for i in range(jumlahDataTraining):
            distanceDataTraining = []
            for j in range(jumlahDataTraining):
                distanceDataTraining.append(self.distance(normalFiturTraining[i], normalFiturTraining[j]))
            
            distance = pd.DataFrame(distanceDataTraining)
            distance = distance.drop(i)
            neighborsDataTraining = self.getNeighbors(distance)
            self.validitasData.append(self.validitas(i, neighborsDataTraining))

    def run(self):
        result = []

        jumlahBarisFiturTesting = len(self.fiturTesting)
        jumlahBarisFiturTraining = len(self.fiturTraining)

        for i in range(jumlahBarisFiturTesting):
            tempData = self.fiturTraining
            tempData.append(self.fiturTesting[i])

            normaltempData = preprocessing.normalize(tempData, norm='l1')
            indexDataTesting = len(normaltempData)-1
            normalDataTesting = normaltempData[indexDataTesting]
            normalDataTraining = np.delete(normaltempData,indexDataTesting,0)
            
            distanceDataTesting= []
            for j in range(jumlahBarisFiturTraining):
                distanceDataTesting.append(self.distance(normalDataTesting, normalDataTraining[j]))
        
            neighborsDataTesting = self.getNeighbors(pd.DataFrame(distanceDataTesting))

            checkLabel = self.isAllLabelSame(neighborsDataTesting)
            if checkLabel['res']:
                result.append(checkLabel['label'])
            else:
                #Vouting
                tempBobot = []
                tempLabel = []

                for k in range(len(neighborsDataTesting)):
                    indexDataTraining = neighborsDataTesting[k]
                    label = self.labelTraining[indexDataTraining]

                    validitasNeighborsDataTesting = self.validitasData[indexDataTraining]
                    distanceNeighborsDataTesting = distanceDataTesting[indexDataTraining]

                    if not label in tempLabel:
                        tempLabel.append(label)
                        tempBobot.append(self.weight(validitasNeighborsDataTesting, distanceNeighborsDataTesting))
                    else:
                        tempBobot[tempLabel.index(label)] += self.weight(validitasNeighborsDataTesting, distanceNeighborsDataTesting)

                result.append(tempLabel[tempBobot.index(max(tempBobot))])
        return result
    
    def isAllLabelSame(self, neighbors):
        result = True
        firstLabel = self.labelTraining[neighbors[0]]

        for i in range(len(neighbors)):
            index = neighbors[i]
            if firstLabel != self.labelTraining[index]:
                result = False

        if result:
            return {'res': True, 'label': firstLabel}
        else:
            return {'res': False}

    def getNeighbors(self, distance):
        vs = distance.nsmallest(self.neighbors, 0)
        return vs.index.values.tolist()

    def distance(self, dataX, dataY):
        if len(dataX) != len(dataY):
            sys.exit("Error when count distance because length not equal")

        temp = 0
        for i in range(len(dataX)):
            temp += pow(dataX[i] - dataY[i], 2)

        return math.sqrt(temp)

    def similarity(self, labelX, labelY):
        if labelX == labelY:
            return 1
        else:
            return 0

    def validitas(self, dataX, neighbors):
        similarityTotal = 0
        for i in range(len(neighbors)):
            similarityTotal += self.similarity(self.labelTraining[dataX], self.labelTraining[neighbors[i]])

        return (1 / self.neighbors) * similarityTotal
    
    def weight(self, validitas, distance):
        return validitas * (1 / (distance + 0.5))