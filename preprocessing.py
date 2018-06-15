import sys
import simplejson as json
import csv
import numpy as np
from gap import gap
import re
import os
import string


numberMatch = re.compile('^[0-9]+$')
printable = set(string.printable)

def processCity(eachJson):
    city = eachJson['city']
    city = re.sub(r'[^\x00-\x7f]', r'', city).replace(" ","").replace("-","").lower()
    eachJson['city'] = city

def preprocess(jsonPath,column_names,preprocessedOutputFile):
    f = open(jsonPath, 'r')
    data = f.read()
    lines = data.split('\n')
    cities=[]
    mapping={}
    newJson = {}
    headersAdded = False
    isOpenindex = 0
    with open(preprocessedOutputFile, 'wb+') as fout:
        csv_file = csv.writer(fout)
        for each in lines:
            if (len(each)>0):
                eachJson = json.loads(each)
                processCity(eachJson)
                city = eachJson['city']
                if eachJson['is_open']!=0 and len(eachJson['postal_code'])>0 and len(city)>0:
                    if not isinstance(eachJson['postal_code'],float):
                        converted = getConvertedZipCode(str(eachJson['postal_code']))
                        eachJson['postal_code'] = converted
                    newJson={}
                    for column in column_names:
                        newJson[column] = eachJson[column]
                    if city not in cities:
                        cities.append(city)
                        mapping[city] = []
                    if not headersAdded:
                        headersAdded = True
                        headers = newJson.keys()
                        isOpenindex = headers.index("is_open")
                        headers.pop(isOpenindex)
                        csv_file.writerow(headers)
                    values = newJson.values()
                    values.pop(isOpenindex)
                    csv_file.writerow(values)
                    mapping[city].append(newJson)
    return mapping,cities


def getConvertedZipCode(postal_code):
    if numberMatch.match(postal_code) is not None:
        return float(postal_code)
    else:
        try:
            val = float(re.sub("\D", "", postal_code))
            if val <= 0:
                val = 1
            while val < 100000:
                val = val * 10
            return val / 10
        except:
            print postal_code

def getBestKValue(zipcodes,maxClusters):
    if maxClusters == 1:
        return maxClusters
    else:
        gaps, s_k, K = gap.gap_statistic(zipcodes, refs=None, B=10, K=range(2,maxClusters), N_init=10)
        if len(gaps) == 0:
            print 1
            return 1
        bestKValue = gap.find_optimal_k(gaps, s_k, K)
        return bestKValue

def clusterLoad(preprocessedCitiesCluster):
    output = {}
    if os.path.isfile(preprocessedCitiesCluster):
        f = open(preprocessedCitiesCluster,'r')
        lines = f.read()
        for line in lines:
            values = line.split(',')
            output[values[0]] = values[1]
    return output

def findClusterValues(data, cities,preprocessedCitiesCluster):
    # variance = 0
    clusters = clusterLoad(preprocessedCitiesCluster)
    processedCities = clusters.keys()
    with open(preprocessedCitiesCluster, 'a') as fout:
        csv_file = csv.writer(fout)
        for eachCity in cities:
            if eachCity != "lasvegas" and eachCity!="toronto" and eachCity not in processedCities:
                businessesInCity = data[eachCity]
                zipcodes = []
                if len(businessesInCity)>1:
                    for eachBusiness in businessesInCity:
                        zipcodes.append(eachBusiness['postal_code'])
                    uniqueLength = len(set(zipcodes))
                    zipcodes = np.array(zipcodes)
                    zipcodes = zipcodes.reshape(-1,1)
                    # totalLength = len(zipcodes)

                    bestK = getBestKValue(zipcodes,uniqueLength)
                    print eachCity + " - "+ str(bestK)
                    clusters[eachCity] = bestK
                    csv_file.writerow([eachCity,bestK])


if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments)> 4 and len(arguments) <1:
        print "Invalid number of arguments - 3 arguments required"
        sys.exit(1)
    jsonPath = arguments[1]
    preprocessedOutputFile = arguments[2]
    preprocessedCitiesCluster = arguments[3]
    column_names = ['city','postal_code','latitude','longitude','is_open','stars']
    mapping,cities = preprocess(jsonPath,column_names,preprocessedOutputFile)
    findClusterValues(mapping,cities,preprocessedCitiesCluster)