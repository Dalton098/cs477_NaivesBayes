import collections
import csv
import math

# Parses the learning data and stores the count for each flower type
#
# Parameters:
# {List of 2D default dicts} trainingData: [0] = setosa, [1] = versicolor, [2] = virginica
# {string} filename: The name of the file that holds the training data
def parseTrainingData(trainingData, fileName):
    with open(fileName) as csvfile:

        setosaCount = 0
        versiCount = 0
        virginicaCount = 0

        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(row[4] == 'Iris-setosa'):
                for j in range(4):
                    trainingData[0][setosaCount][j] = row[j]
                setosaCount += 1
            elif(row[4] == 'Iris-versicolor'):
                for j in range(4):
                    trainingData[1][versiCount][j] = row[j]
                versiCount += 1
            elif(row[4] == "Iris-virginica"):
                for j in range(4):
                    trainingData[2][virginicaCount][j] = row[j]
                virginicaCount += 1

# Parses the test data (provided in csv) which will all be classified
#
# Parameters:
# {string} fileName: The name of the file where the test data is stored
#
# Returns a 2D default dictionary containing the test data
# where the row is the entry and the columns are the attributes
def parseTestData(fileName):
    with open(fileName) as csvfile:

        toReturn = collections.defaultdict(
            lambda: collections.defaultdict(float))
        count = 0

        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            for i in range(4):
                toReturn[count][i] = row[i]
            count += 1

    return toReturn

# Takes in the mean of all the data in the given column
# and then categorizes the current data point into one of the categories
# 1: 0 -> 1/2(mean)
# 2: 1/2(mean) -> mean
# 3: mean -> 3/2(mean)
# 4: 3/2(mean) -> infinity
#
# Parameters:
# {2D Default Dict} dict: The data to be categorized
# {int} i: The row in the dictionary
# {int} j: The column in the dictionary
# {float} mean: The mean (from the training data) of the column
# {float} halfMean: Half of the mean of the column
# {float} threeHalfMean: 3/2 of the mean of the column
def categorize(dict, i, j, mean, halfMean, threeHalfMean):
    toChange = float(dict[i][j])
    if(0 <= toChange and toChange < halfMean):
        dict[i][j] = 1
    elif (halfMean <= toChange and toChange < mean):
        dict[i][j] = 2
    elif (mean <= toChange and toChange < threeHalfMean):
        dict[i][j] = 3
    else:
        dict[i][j] = 4

# Sets up four categories (1, 2, 3, 4) for each column to make the data discrete
#
# Parameters:
# {List of 2D default dicts} trainingData: [0] = setosa, [1] = versicolor, [2] = virginica
#
# Returns a list containing the means of each column which will be used in the classification step
# return: [0] = Sepal Length Mean, [1] = Sepal Width Mean, [2] = Petal Length Mean, [3] = Petal Width Mean
def makeDiscrete(trainingData):

    toReturn = []
    totalEntries = len(trainingData[0]) + \
        len(trainingData[1]) + len(trainingData[2])

    # 4 for the 4 columns
    for j in range(4):
        mean = 0.0
        for k in range(len(trainingData)):

            for i in range(len(trainingData[k])):
                mean += float(trainingData[k][i][j])

        mean = mean/totalEntries
        toReturn.append(mean)
        halfMean = mean/2
        threeHalfMean = (3/2 * mean)

        for k in range(len(trainingData)):
            for i in range(len(trainingData[k])):
                categorize(trainingData[k], i, j, mean,
                           halfMean, threeHalfMean)

    return toReturn

# Calculates the probabilities for all the categories in the training data
#
# {List of 2D default dict} trainingData: trainingData: [0] = setosa, [1] = versicolor, [2] = virginica
#
# Returns a list containing 2D default dictionaries
# The depth determines which plant type, the column determines the attribute and the row determines the category
# ie: [0][0][0] = Probability of the sepal length being in category 1 for Iris-setosa
# ie: [1][3][2] = Probability of the petal length being in category 4 for Iris-versicolor
def calculateProbabilities(trainingData):
    probabilities = [collections.defaultdict(lambda: collections.defaultdict(float)), collections.defaultdict(
        lambda: collections.defaultdict(float)), collections.defaultdict(lambda: collections.defaultdict(float))]

    for k in range(len(trainingData)):
        for j in range(4):

            # Counts of each category per column per plant
            countOfOne = 0
            countOfTwo = 0
            countOfThree = 0
            countOfFour = 0

            for i in range(len(trainingData[k])):
                toCount = trainingData[k][i][j]

                if (toCount == 1):
                    countOfOne += 1
                elif (toCount == 2):
                    countOfTwo += 1
                elif (toCount == 3):
                    countOfThree += 1
                elif (toCount == 4):
                    countOfFour += 1

            # Adding 1 for smoothing  (Not true smoothing more so because cant take the log of 0)
            entryAmount = len(trainingData[k])
            countOfOne = (countOfOne + 1) / entryAmount
            countOfTwo = (countOfTwo + 1) / entryAmount
            countOfThree = (countOfThree + 1) / entryAmount
            countOfFour = (countOfFour + 1) / entryAmount

            probabilities[k][0][j] = countOfOne
            probabilities[k][1][j] = countOfTwo
            probabilities[k][2][j] = countOfThree
            probabilities[k][3][j] = countOfFour

    return probabilities

# Classifies the given data to one of the plant types using Naive Bayes
#
# Parameters:
# {2D Default Dict} toClassify: The test data to classify into a plant type
# {List of 2D default dicts} trainingData: [0] = setosa, [1] = versicolor, [2] = virginica
# {list} means: List containing the means (from the training data) of all the different plant attributes
def classify(toClassify, trainingData, means):

    toReturn = []
    totalEntries = len(trainingData[0]) + \
        len(trainingData[1]) + len(trainingData[2])

    priorSetosa = len(trainingData[0]) / totalEntries
    priorVersi = len(trainingData[1]) / totalEntries
    priorVirginica = len(trainingData[2]) / totalEntries

    probabilities = calculateProbabilities(trainingData)

    for j in range(4):

        mean = means[j]
        halfMean = mean / 2
        threeHalfMean = (3/2 * mean)

        for i in range(len(toClassify)):
            categorize(toClassify, i, j, mean, halfMean, threeHalfMean)

    for i in range(len(toClassify)):
        temp = toClassify[i]

        indexSepalLength = temp[0] - 1
        indexSepalWidth = temp[1] - 1
        indexPetalLength = temp[2] - 1
        indexPetalWidth = temp[3] - 1

        probSetosa = math.log(priorSetosa) + math.log(probabilities[0][indexSepalLength][0]) + math.log(
            probabilities[0][indexSepalWidth][1]) + math.log(probabilities[0][indexPetalLength][2]) + math.log(probabilities[0][indexPetalWidth][3])
        probVersi = math.log(priorVersi) + math.log(probabilities[1][indexSepalLength][0]) + math.log(
            probabilities[1][indexSepalWidth][1]) + math.log(probabilities[1][indexPetalLength][2]) + math.log(probabilities[1][indexPetalWidth][3])
        probVirginica = math.log(priorVirginica) + math.log(probabilities[2][indexSepalLength][0]) + math.log(
            probabilities[2][indexSepalWidth][1]) + math.log(probabilities[2][indexPetalLength][2]) + math.log(probabilities[2][indexPetalWidth][3])

        if (probSetosa >= probVersi and probSetosa >= probVirginica):
            toReturn.append('Iris-setosa')
        elif (probVersi >= probSetosa and probVersi >= probVirginica):
            toReturn.append('Iris-versicolor')
        else:
            toReturn.append('Iris-virginica')

    return toReturn

def prettyPrint(results):
    for i in range(len(results)):
        print(str(i) + ':\t' + results[i])

def prettyPrint2(toPrint):
    for row in toPrint:
        print(toPrint[row])

# Note: For all tables
# Column 0: Sepal Length
# Column 1: Sepal Width
# Column 2: Petal Length
# Column 3: Petal Width
setosaDict = collections.defaultdict(lambda: collections.defaultdict(float))
versicolorDict = collections.defaultdict(
    lambda: collections.defaultdict(float))
virginicaDict = collections.defaultdict(lambda: collections.defaultdict(float))

trainingData = []
trainingData.append(setosaDict)
trainingData.append(versicolorDict)
trainingData.append(virginicaDict)

parseTrainingData(trainingData, 'IRIS.csv')
means = makeDiscrete(trainingData)

testData = parseTestData('IRIS.csv')

results = classify(testData, trainingData, means)
prettyPrint(results)
