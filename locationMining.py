import sys
import csv
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import re
from sklearn import tree
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

numberMatch = re.compile('^[0-9]+$')

def loadFile(preprocessedFilePath):
    with open(preprocessedFilePath,'rb') as f:
        data = csv.DictReader(f)
        cities=[]
        mapping={}
        for each in data:
            if (len(each)>0):
                city = each['city']
                if city not in cities:
                    cities.append(city)
                    mapping[city] = []
                for key in each.keys():
                    if key!='city':
                        each[key]=float(each[key])
                mapping[city].append(each)
    return mapping,cities



def clusterLoad(preprocessedCitiesCluster):
   f = open(preprocessedCitiesCluster)
   citiesCurrent=[]
   CityCluster = {}
   lines = f.read().split('\r\n')
   for line in lines:
       if (len(line) > 0):
           values = line.split(',')
           CityCluster[values[0]] = int(values[1])
           citiesCurrent.append(values[0])
   return CityCluster,citiesCurrent

def kMeansClustering(data, city, cityCluster):
    businessesInCity = data[city]
    zipcodes = []
    for eachBusiness in businessesInCity:
        zipcodes.append(eachBusiness['postal_code'])
    totalLength = len(zipcodes)
    k = cityCluster
    zipcodes = np.array(zipcodes)
    zipcodes = zipcodes.reshape(-1, 1)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(zipcodes)
    labels = kmeans.predict(zipcodes)
    zipCodeMapping = {}
    for i in range(totalLength):
        if not zipCodeMapping.has_key(labels[i]):
            zipCodeMapping[labels[i]] = {}
            zipCodeMapping[labels[i]]['businesses'] = []
            zipCodeMapping[labels[i]]['stars'] = []
            zipCodeMapping[labels[i]]['variance'] = 0.0
            zipCodeMapping[labels[i]]['mean'] = 0.0
        zipCodeMapping[labels[i]]['businesses'].append(businessesInCity[i])
        zipCodeMapping[labels[i]]['stars'].append(businessesInCity[i]['stars']*2)
    for i in range(k):
        var = np.var(zipCodeMapping[i]['stars'])
        if math.isnan(var):
            var = 0
        zipCodeMapping[i]['variance'] = var
        zipCodeMapping[i]['mean'] = np.mean(zipCodeMapping[i]['stars'])
    return {"type": kmeans, "mapping": zipCodeMapping}

def aWardClustering(data, city, cityCluster):
    businessesInCity = data[city]
    zipcodes = []
    for eachBusiness in businessesInCity:
        zipcodes.append(eachBusiness['postal_code'])
    k = cityCluster
    zipcodes = np.array(zipcodes)
    zipcodes = zipcodes.reshape(-1, 1)
    totalLength = len(zipcodes)
    ward = AgglomerativeClustering(n_clusters=k).fit(zipcodes)
    labels = ward.labels_
    zipCodeMapping = {}
    for i in range(totalLength):
        if not zipCodeMapping.has_key(labels[i]):
            zipCodeMapping[labels[i]] = {}
            zipCodeMapping[labels[i]]['businesses'] = []
            zipCodeMapping[labels[i]]['stars'] = []
            zipCodeMapping[labels[i]]['variance'] = 0
            zipCodeMapping[labels[i]]['mean'] = 0
        zipCodeMapping[labels[i]]['businesses'].append(businessesInCity[i])
        zipCodeMapping[labels[i]]['stars'].append(businessesInCity[i]['stars'])
    for i in range(k):
        zipCodeMapping[labels[i]]['variance'] = np.var(zipCodeMapping[labels[i]]['stars'])
        zipCodeMapping[labels[i]]['mean'] = np.mean(zipCodeMapping[labels[i]]['stars'])
    return {"type": ward, "mapping": zipCodeMapping}

def aCompleteClustering(data, city, cityCluster):
    businessesInCity = data[city]
    zipcodes = []
    for eachBusiness in businessesInCity:
        zipcodes.append(eachBusiness['postal_code'])
    k = cityCluster
    zipcodes = np.array(zipcodes)
    zipcodes = zipcodes.reshape(-1, 1)
    totalLength = len(zipcodes)
    complete = AgglomerativeClustering(n_clusters=k,linkage='complete').fit(zipcodes)
    labels = complete.labels_
    zipCodeMapping = {}
    for i in range(totalLength):
        if not zipCodeMapping.has_key(labels[i]):
            zipCodeMapping[labels[i]] = {}
            zipCodeMapping[labels[i]]['businesses'] = []
            zipCodeMapping[labels[i]]['stars'] = []
            zipCodeMapping[labels[i]]['variance'] = 0
            zipCodeMapping[labels[i]]['mean'] = 0
        zipCodeMapping[labels[i]]['businesses'].append(businessesInCity[i])
        zipCodeMapping[labels[i]]['stars'].append(businessesInCity[i]['stars'])
    for i in range(k):
        zipCodeMapping[labels[i]]['variance'] = np.var(zipCodeMapping[labels[i]]['stars'])
        zipCodeMapping[labels[i]]['mean'] = np.mean(zipCodeMapping[labels[i]]['stars'])
    return {"type": complete, "mapping": zipCodeMapping}

def aAverageClustering(data, city, cityCluster):
    businessesInCity = data[city]
    zipcodes = []
    for eachBusiness in businessesInCity:
        zipcodes.append(eachBusiness['postal_code'])
    k = cityCluster
    zipcodes = np.array(zipcodes)
    zipcodes = zipcodes.reshape(-1, 1)
    totalLength = len(zipcodes)
    average = AgglomerativeClustering(n_clusters=k, linkage='average').fit(zipcodes)
    labels = average.labels_
    zipCodeMapping = {}
    for i in range(totalLength):
        if not zipCodeMapping.has_key(labels[i]):
            zipCodeMapping[labels[i]] = {}
            zipCodeMapping[labels[i]]['businesses'] = []
            zipCodeMapping[labels[i]]['stars'] = []
            zipCodeMapping[labels[i]]['variance'] = 0
            zipCodeMapping[labels[i]]['mean'] = 0
        zipCodeMapping[labels[i]]['businesses'].append(businessesInCity[i])
        zipCodeMapping[labels[i]]['stars'].append(businessesInCity[i]['stars'])
    for i in range(k):
        zipCodeMapping[labels[i]]['variance'] = np.var(zipCodeMapping[labels[i]]['stars'])
        zipCodeMapping[labels[i]]['mean'] = np.mean(zipCodeMapping[labels[i]]['stars'])
    return {"type": average, "mapping": zipCodeMapping}

def dbScanClustering(data, city, cityCluster):
    businessesInCity = data[city]
    zipcodes = []
    for eachBusiness in businessesInCity:
        zipcodes.append(eachBusiness['postal_code'])
    k = cityCluster
    zipcodes = np.array(zipcodes)
    zipcodes = zipcodes.reshape(-1, 1)
    totalLength = len(zipcodes)
    dbScan = DBSCAN(eps=0.5).fit(zipcodes)
    labels = dbScan.labels_
    zipCodeMapping = {}
    for i in range(totalLength):
        if not zipCodeMapping.has_key(labels[i]):
            zipCodeMapping[labels[i]] = {}
            zipCodeMapping[labels[i]]['businesses'] = []
            zipCodeMapping[labels[i]]['stars'] = []
        zipCodeMapping[labels[i]]['businesses'].append(businessesInCity[i])
        zipCodeMapping[labels[i]]['stars'].append(businessesInCity[i]['stars'])
    for i in range(k):
        zipCodeMapping[labels[i]]['variance'] = np.var(zipCodeMapping[labels[i]]['stars'])
        zipCodeMapping[labels[i]]['mean'] = np.mean(zipCodeMapping[labels[i]]['stars'])
    return {"type":dbScan,"mapping":zipCodeMapping}

def zipcodeclustering(mapping, currentCity, CityCluster):
    clusters = []
    clusters.append(kMeansClustering(mapping, currentCity, CityCluster))
    # clusters.append(aWardClustering(mapping, currentCity, CityCluster))
    # clusters.append(aCompleteClustering(mapping, currentCity, CityCluster))
    # clusters.append(aAverageClustering(mapping, currentCity, CityCluster))
    # clusters.append(dbScanClustering(mapping, currentCity, CityCluster))
    return clusters

def getBestAccuracy(estimators,scaled_data,classValues,sd,fout,clusterNumber):
    try:
        totallen = len(scaled_data)
        leng = int(totallen * 0.7)
        maxAccuracy = 0
        minAccuracy = 100
        index = 0
        for clf in estimators:
            index+=1
            clf.fit(scaled_data[:leng],classValues[:leng])
            trueValues = classValues[leng:]
            predictedValues = clf.predict(scaled_data[leng:])
            correct = 0
            wrong = 0
            for i in range(len(trueValues)):
                if predictedValues[i]>(trueValues[i] - sd) and predictedValues[i]<(trueValues[i] + sd):
                    correct+=1.0
                else:
                    wrong+=1.0
            accuracy = correct/(correct+wrong)
            if maxAccuracy < accuracy:
                maxAccuracy = accuracy
            if minAccuracy>accuracy:
                minAccuracy = accuracy
        fout.write("The true values of the cluster varies with the predicted values of the model with +/-"+str(sd)+
                   " with an accuracy of %"+str(maxAccuracy*100)+".\n")
        if maxAccuracy>0.50:
            fout.write("The success of the businesses in this cluster depends only on the location\n")
        else:
            fout.write("The success of the businesses in this cluster does not depend only on the location\n")
        return maxAccuracy,minAccuracy
    except:
        return 0,0

def comparison(comp_list,outputFilePath):
    f = open(outputFilePath, 'w')
    m = max(comp_list)
    if m == comp_list[0]:
        f.write("decision tree is the best, with accuracy: %0.2f " %comp_list[0])

def findBestClassifier(clf,clusters,city,outputFile):
    size = len(clusters)
    n = 0
    isCityAdded = False
    for i in range(size):
        cluster = clusters[i]
        scaled_data = []
        classValues = []
        b_size = len(cluster["businesses"])
        is_first_time = True
        city_index = 0
        stars_index = 0
        for index in range(b_size):
            business = cluster["businesses"][index]
            values = business.values()
            if is_first_time:
                is_first_time = False
                headers = business.keys()
                city_index = headers.index("city")
                stars_index = headers.index("stars")
            if city_index < stars_index:
                classValues.append(values.pop(stars_index)*2)
                values.pop(city_index)
            else:
                values.pop(city_index)
                classValues.append(values.pop(stars_index)*2)
            scaled_data.append(values)

        if len(classValues) > 5:
            n +=1
            if not isCityAdded:
                isCityAdded = True
                fout.write("For City - " + str(city) + "\n")
            outputFile.write("For Cluster "+str(n)+"\n")
            outputFile.write("Mean of stars in the cluster is " + str(cluster['mean']/2) + "\n")
            outputFile.write("Variance of stars in the cluster is " + str(cluster['variance']/4) + "\n")
            maxAccuracy, minAccuracy = getBestAccuracy(clf,scaled_data, classValues,math.sqrt(cluster['variance']/4),outputFile,n)
            # print maxAccuracy,minAccuracy
            outputFile.write("\n\n")

def getEstimators():
    clf = []
    clf.append(tree.DecisionTreeClassifier(criterion='entropy'))
    clf.append(Perceptron(n_iter=500, random_state=15, fit_intercept=True, eta0=0.1))
    clf.append(MLPClassifier(solver='sgd', activation='logistic',
                             hidden_layer_sizes=(500),
                             learning_rate_init=0.01,
                             early_stopping=False, max_iter=1000,
                             random_state=1))
    clf.append(SVC(kernel='rbf'))
    clf.append(LogisticRegression())
    clf.append(KNeighborsClassifier(n_neighbors=1))
    clf.append(RandomForestClassifier(n_estimators=30, max_depth=None, min_samples_split=2, random_state=0))
    clf.append(AdaBoostClassifier(n_estimators=50))
    return clf

if __name__ == "__main__":
    arguments = sys.argv
    if len(arguments)> 4 and len(arguments) <1:
        print "Invalid number of arguments - 4 arguments required"
        sys.exit(1)
    dataPath = arguments[1]
    clusterFilePath = arguments[2]
    outputFilePath=arguments[3]
    mapping, cities = loadFile(dataPath)
    CityCluster, citiesCurrent = clusterLoad(clusterFilePath)
    clf = getEstimators()
    with open(outputFilePath, 'w') as fout:
        for eachCity in citiesCurrent:
            clusters = zipcodeclustering(mapping, eachCity, CityCluster[eachCity])
            findBestClassifier(clf,clusters[0]["mapping"],eachCity,fout)