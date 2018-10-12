import scipy.io.wavfile
import numpy as np
import math


# Return the names of the files
def getFileNames():
    musicSpeech = open("music_speech.mf")
    fileNames = []
    for fileNameAndLabel in musicSpeech.read().split("\n"):
        fileNames.append(fileNameAndLabel.replace("\tmusic", "").replace("\tspeech", ""))
    return fileNames


'''
Get the buffers matrix :
1/ Load a wav file
2/ Convert the data to floats
3/ Split the data into buffers of length 1024 with 50% overlap
'''
def getBuffersMatrix(fileName):
    data = scipy.io.wavfile.read("music_speech/" + fileName)[1]
    data = np.divide(data, 32768)
    buffersNumber = math.floor(len(data)/512) - 1 # The total number of buffers to create
    buffersMatrix = np.empty([buffersNumber, 1024]) # We create an empty matrix which will contain the buffers
    for i in range(buffersNumber):
        buffersMatrix[i,0:1024] = data[512*i:512*i+1024] # We fill the matrix with the buffers data
    return buffersMatrix


# Functions calculating the 5 features
def rootMeanSquared(data):
    return math.sqrt(np.sum(np.square(data))/len(data))


def peakToAverageRatio(data):
    return round(np.amax(np.absolute(data))/rootMeanSquared(data), 6)


def zeroCrossings(data):
    data1 = np.delete(data, 0)
    data2 = np.delete(data, -1)
    '''
    1/ np.multiply(data1, data2) gives us xi * xi-1
    2/ np.sign converts all negative values obtained to -1 and all positive values to 1. 0 stays 0.
    3/ np.clip(..., -1, 0) converts all 1 (got when xi * xi-1 > 0) to 0
    4/ We multiply the result by -1 in order to convert all -1 (got when xi * xi-1 < 0) to 1.
    5/ Then we can sum, divide by N-1, and round the result
    '''
    return round(np.sum(np.clip(np.sign(np.multiply(data1, data2)), -1, 0)*(-1))/(len(data) - 1), 6)


def medianAbsoluteDeviation(data):
    return round(np.median(np.absolute(np.subtract(data, np.median(data)))), 6)


def meanAbsoluteDeviation(data):
    return round(np.mean(np.absolute(np.subtract(data, np.mean(data)))), 6)


# Get the features of a file split into buffers
def getFeatures(fileName):
    buffersMatrix = getBuffersMatrix(fileName)
    buffersNumber = buffersMatrix.shape[0]
    featuresMatrix = np.empty([buffersNumber, 5]) # We create an empty matrix which will contain the features
    for i in range(buffersNumber):
        buffer = buffersMatrix[i, 0:1024]
        # We fill the matrix with the features
        featuresMatrix[i, 0:5] = [round(rootMeanSquared(buffer), 6), peakToAverageRatio(buffer), zeroCrossings(buffer), medianAbsoluteDeviation(buffer), meanAbsoluteDeviation(buffer)]
    return featuresMatrix


# Computes the mean and standard deviation of a matrix M along the axis 0
def meanAndStd(M): 
    return np.concatenate([np.mean(M, axis=0), np.std(M, axis=0)])


# Output the data to an ARFF file
def createArffFile():
    with open("musicSpeechData2.arff", "w") as musicSpeechData2:
        musicSpeechData2.write('''@RELATION music_speech
@ATTRIBUTE RMS_MEAN NUMERIC
@ATTRIBUTE PAR_MEAN NUMERIC
@ATTRIBUTE ZCR_MEAN NUMERIC
@ATTRIBUTE MAD_MEAN NUMERIC
@ATTRIBUTE MEAN_AD_MEAN NUMERIC
@ATTRIBUTE RMS_STD NUMERIC
@ATTRIBUTE PAR_STD NUMERIC
@ATTRIBUTE ZCR_STD NUMERIC
@ATTRIBUTE MAD_STD NUMERIC
@ATTRIBUTE MEAN_AD_STD NUMERIC
@ATTRIBUTE class {music,speech}
@DATA\n''') # Write the header of the ARFF file
        for fileName in getFileNames():
            musicSpeechData2.write(",".join(str(x) for x in meanAndStd(getFeatures(fileName))) + ",") # Write the features (converted to strings) of the file
            musicSpeechData2.write(fileName.split("_")[0]+"\n") # Add the label of the file
    

createArffFile()
