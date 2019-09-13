import collections
import csv

# Parses the dataset into 3 different 2d defaultdicts


def parseData(setosa, versicolor, virginica):
    with open('IRIS.csv') as csvfile:

        setosaCount = 0
        versiCount = 0
        virginicaCount = 0

        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if(row[4] == 'Iris-setosa'):
                setosa[setosaCount][0] = row[0]
                setosa[setosaCount][1] = row[1]
                setosa[setosaCount][2] = row[2]
                setosa[setosaCount][3] = row[3]
                setosaCount = setosaCount + 1
            elif(row[4] == 'Iris-versicolor'):
                versicolor[versiCount][0] = row[0]
                versicolor[versiCount][1] = row[1]
                versicolor[versiCount][2] = row[2]
                versicolor[versiCount][3] = row[3]
                versiCount = versiCount + 1
            elif(row[4] == "Iris-virginica"):
                virginica[virginicaCount][0] = row[0]
                virginica[virginicaCount][1] = row[1]
                virginica[virginicaCount][2] = row[2]
                virginica[virginicaCount][3] = row[3]
                virginicaCount = virginicaCount + 1
    return [setosaCount, versiCount, virginicaCount, setosaCount + versiCount + virginicaCount, ]

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

# Parses the data and stores the count for each flower type
# These counts will be used later in the naive bayes calcuation
counts = parseData(setosaDict, versicolorDict, virginicaDict)

