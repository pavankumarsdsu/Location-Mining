import re
import sys
import random

ITERATIONS = 25

def loadFile(inputFileName):
    f = open(inputFileName, 'r')
    data = f.read()
    lines = re.split(r'\r', data)
    instances = []
    isHeader = True
    headers = []
    headerCount = 0
    for eachLine in lines:
        if isHeader:
            isHeader = False
            headers = re.split(r'\t',eachLine)
            headerCount = len(headers)
        else:
            newInstance = {}
            values = re.split(r'\t',eachLine)
            for i in range(headerCount): #id values are not required
                newInstance[headers[i]] = float(values[i])
            newInstance['euclideanDistance'] = 0.0
            newInstance['Cluster'] = 1
            instances.append(newInstance)
    return instances

def cluster(data,clusterCount,outputFileName):
    instanceCount = len(data)
    if (clusterCount>=instanceCount or clusterCount<=0):
        print "Number of clusters should be greater than 0 and less than the sample size"
        sys.exit(1)
    randomIndexes = random.sample(range(instanceCount),clusterCount)
    centroids = []
    headers = []
    isHeadersAdded = False
    SSE = 0
    for eachIndex in randomIndexes:
        centroid = dict(data[eachIndex])
        centroid.pop('id')
        centroid.pop('Cluster')
        centroid.pop('euclideanDistance')
        if not isHeadersAdded:
            headers.extend(centroid.keys())
            isHeadersAdded = True
        centroids.append(centroid)
    for iteration in range(ITERATIONS):
        for instanceIndex in range(instanceCount):
            eachInstance = data[instanceIndex]
            minSum = 0
            isSumInitialized = False
            for clusterIndex in range(clusterCount):
                sum = 0
                eachCentroid = centroids[clusterIndex]
                for eachHeader in headers:
                    temp = (eachCentroid[eachHeader] - eachInstance[eachHeader])
                    sum += (temp*temp)
                if not isSumInitialized:
                    sum = sum
                    minSum = sum
                    isSumInitialized = True
                if (sum<minSum):
                    minSum = sum
                    data[instanceIndex]['euclideanDistance'] = sum
                    data[instanceIndex]['Cluster'] = clusterIndex+1
        SSE = getSSE(data, centroids,clusterCount)
        centroids[:] = []
        newCentroids = getNewCentroids(data, clusterCount)
        if newCentroids is not None:
            centroids.extend(newCentroids)
    printOutput(outputFileName,data,clusterCount,SSE)

def printOutput(outputFileName,data,clusterCount,SSE):
    clusters = []
    for index in range(clusterCount):
        cluster={}
        cluster['index'] = index+1
        cluster['values'] = []
        clusters.append(cluster)
    for eachInstance in data:
        index = eachInstance['Cluster']-1
        clusters[index]['values'].append(int(eachInstance['id']))
    with open(outputFileName, 'wb+') as fout:
        for eachCluster in clusters:
            string = str(eachCluster['index'])
            string +=" : "+str(eachCluster['values'])[1:-1]+"\n"
            fout.write(string)
        fout.write(" Squared Sum Error (SSE) = "+str(SSE))

def getSSE(data, centroids,clusterCount):
    instanceCount = len(data)
    headers = centroids[0].keys()
    SSE = 0
    for instanceIndex in range(instanceCount):
        eachInstance = data[instanceIndex]
        minSum = 0
        isSumInitialized = False
        for clusterIndex in range(clusterCount):
            sum = 0
            eachCentroid = centroids[clusterIndex]
            for eachHeader in headers:
                temp = (eachCentroid[eachHeader] - eachInstance[eachHeader])
                sum = (temp * temp)
            if not isSumInitialized:
                sum = sum
                minSum = sum
                isSumInitialized = True
            if (sum < minSum):
                minSum = sum
        SSE += minSum
    return SSE

def getNewCentroids(data,clusterCount):
    centroids = []
    tempData = dict(data[0])
    tempData.pop('Cluster')
    tempData.pop('id')
    tempData.pop('euclideanDistance')
    keys = tempData.keys()
    for i in range(clusterCount):
        centroid = {}
        keys = tempData.keys()
        for eachKey in keys:
            centroid[eachKey] = 0.0
        centroid['clusterCount'] = 0
        centroids.append(centroid)
    for eachInstance in data:
        index = eachInstance['Cluster']-1
        for eachKey in keys:
            centroids[index][eachKey] += eachInstance[eachKey]
        centroids[index]['clusterCount']+=1
    for clusterIndex in range(clusterCount):
        count = centroids[clusterIndex]['clusterCount']
        for eachKey in keys:
            centroids[clusterIndex][eachKey] /= count
        centroids[clusterIndex].pop('clusterCount')
    return centroids

if __name__ == "__main__":
    if (len(sys.argv) == 4):
        numberOfClusters = int(sys.argv[1])
        inputFileName = sys.argv[2]
        outputFileName = sys.argv[3]
        data = loadFile(inputFileName)
        cluster(data,numberOfClusters,outputFileName)
    else:
        print "Invalid Input arguments"
        print "python.exe K-Means.py numberOfClusters inputFileName outputFileName"
