import math
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.io.wavfile


# Return the names of the files
def getFileNames():
    musicSpeech = open("music_speech.mf")
    fileNames = []
    for fileNameAndLabel in musicSpeech.read().split("\n"):
        fileNames.append(fileNameAndLabel.replace("\tmusic", "").replace("\tspeech", ""))
    return fileNames


# Return the spectral buffers for a given file
def getSpectralBuffers(fileName):
    data = scipy.io.wavfile.read("music_speech/" + fileName)[1] # Load the wav file
    data = np.divide(data, 32768) # Convert the data to floats
    bufferLength = 1024
    buffersNumber = math.floor(len(data)/512) - 1 # The total number of buffers to create
    buffersMatrix = np.empty([buffersNumber, math.floor(bufferLength/2) + 1]) # Create an empty matrix which will contain the spectral buffers
    for i in range(buffersNumber):
        fft = scipy.fftpack.fft(scipy.signal.hamming(bufferLength)*data[512*i:512*i+bufferLength]) # Split the data into buffers, mutiply each buffer by a Hamming window, and take the fft
        buffersMatrix[i] = np.abs(fft[:math.floor(len(fft)/2) + 1]) # Fill the matrix with the absolute value of the spectral buffers
    return buffersMatrix


# Functions calculating the 5 features
def spectralCentroid(data):
    return np.sum(np.multiply(np.arange(len(data)), data))/np.sum(data)


def spectralRollOff(data, L):
    threshold = L*np.sum(data)
    return np.where(np.cumsum(data) > threshold)[0][0]


def spectralFlatnessMeasure(data):
    geometricMean = np.exp((1/len(data))*np.sum(np.log(data)))
    arithmeticMean = (1/len(data))*np.sum(data)
    return geometricMean/arithmeticMean

 
def peakToAverageRatio(data):
    rootMeanSquared = math.sqrt(np.sum(np.square(data))/len(data))
    return np.amax(data)/rootMeanSquared


def spectralFlux(data1, data2):
    diff = np.subtract(data2, data1)
    return np.sum(diff[diff > 0]) # Sum only the positive values of diff


# Get the features of a file split into buffers
def getFeatures(fileName):
    buffersMatrix = getSpectralBuffers(fileName)
    buffersNumber = buffersMatrix.shape[0]
    featuresMatrix = np.empty([buffersNumber, 5]) # Create an empty matrix which will contain the features
    for i in range(buffersNumber):
        buffer = buffersMatrix[i]
        previousBuffer = buffersMatrix[i-1] if i!=0 else np.zeros(buffersMatrix.shape[1]) # Used to compute the spectral flux
        featuresMatrix[i] = [spectralCentroid(buffer), spectralRollOff(buffer, 0.85), spectralFlatnessMeasure(buffer), peakToAverageRatio(buffer), spectralFlux(previousBuffer, buffer)]
    return featuresMatrix


# Computes the mean and standard deviation of a matrix M along the axis 0
def meanAndStd(M): 
    return np.concatenate([np.mean(M, axis=0), np.std(M, axis=0)])


# Output the data to an ARFF file
def createArffFile():
    with open("frequencyDomainFeatures.arff", "w") as frequencyDomainFeatures:
        frequencyDomainFeatures.write('''@RELATION music_speech
@ATTRIBUTE SC_MEAN NUMERIC
@ATTRIBUTE SRO_MEAN NUMERIC
@ATTRIBUTE SFM_MEAN NUMERIC
@ATTRIBUTE PARFFT_MEAN NUMERIC
@ATTRIBUTE FLUX_MEAN NUMERIC
@ATTRIBUTE SC_STD NUMERIC
@ATTRIBUTE SRO_STD NUMERIC
@ATTRIBUTE SFM_STD NUMERIC
@ATTRIBUTE PARFFT_STD NUMERIC
@ATTRIBUTE FLUX_STD NUMERIC
@ATTRIBUTE class {music,speech}
@DATA\n''') # Write the header of the ARFF file
        for fileName in getFileNames():
            frequencyDomainFeatures.write(",".join(str(x) for x in meanAndStd(getFeatures(fileName))) + ",") # Write the features (converted to strings) of the file
            frequencyDomainFeatures.write(fileName.split("_")[0]+"\n") # Add the label of the file
    

createArffFile()
