import collections
import csv

# Parses the data and stores the count for each flower type
# Returns counts that will be used later in the naive bayes calcuation
# [0] = setosaCount, [1] = versiCount, [2] = virginicaCount, [3], totalCount


def parseData(setosa, versicolor, virginica):
    with open('IRIS.csv') as csvfile:

        setosaCount = 0
        versiCount = 0
        virginicaCount = 0

        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(row[4] == 'Iris-setosa'):
                for i in range(4):
                    setosa[setosaCount][i] = row[i]
                setosaCount += 1
            elif(row[4] == 'Iris-versicolor'):
                for i in range(4):
                    versicolor[versiCount][i] = row[i]
                versiCount += 1
            elif(row[4] == "Iris-virginica"):
                for i in range(4):
                    virginica[virginicaCount][i] = row[i]
                virginicaCount += 1
    return [setosaCount, versiCount, virginicaCount, setosaCount + versiCount + virginicaCount, ]

# Takes in the mean of all the data in the given column
# and then categorizes the current data point into one of the categories
# 1: 0 -> 1/2(mean)
# 2: 1/2(mean) -> mean
# 3: mean -> 3/2(mean)
# 4: 3/2(mean) -> infinity


def categorizeData(dict, i, j, mean, halfMean, threeHalfMean):
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


def makeDiscrete(setosa, versicolor, virginica, counts):
    toReturn = []
    # 4 for the 4 columns
    for j in range(4):
        mean = 0.0
        for i in range(counts[0]):
            mean += float(setosa[i][j])
        for i in range(counts[1]):
            mean += float(versicolor[i][j])
        for i in range(counts[2]):
            mean += float(virginica[i][j])

        mean = mean/counts[3]
        toReturn.append(mean)
        halfMean = mean/2
        threeHalfMean = (3/2 * mean)

        for i in range(counts[0]):
            categorizeData(setosa, i, j, mean, halfMean, threeHalfMean)
        for i in range(counts[1]):
            categorizeData(versicolor, i, j, mean, halfMean, threeHalfMean)
        for i in range(counts[2]):
            categorizeData(virginica, i, j, mean, halfMean, threeHalfMean)

    return toReturn


# DELETE THIS EVENTUALLY
def prettyPrint(toPrint):
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

counts = parseData(setosaDict, versicolorDict, virginicaDict)
means = makeDiscrete(setosaDict, versicolorDict, virginicaDict, counts)
