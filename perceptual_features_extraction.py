import math
import numpy as np
import scipy.signal
import scipy.fftpack
import scipy.io.wavfile
import matplotlib as plt


# Some constants used in the code
nyquistF = 11025
bufferLength = 1024
fftLength = math.floor(bufferLength / 2)
filtersNumber = 26


# Return the names of the files
def getFileNames():
    musicSpeech = open("music_speech.mf")
    fileNames = []
    for fileNameAndLabel in musicSpeech.read().split("\n"):
        fileNames.append(fileNameAndLabel.replace("\tmusic", "").replace("\tspeech", ""))
    return fileNames


# Return a buffer matrix for a given file
def getBuffersMatrix(fileName):
    data = scipy.io.wavfile.read("music_speech/" + fileName)[1] # Load the wav file
    data = np.divide(data, 32768) # Convert the data to floats
    buffersNumber = math.floor(len(data)/512) - 1 # The total number of buffers to create
    buffersMatrix = np.empty([buffersNumber, bufferLength]) # Create an empty matrix which will contain the buffers
    for i in range(buffersNumber):
        buffersMatrix[i] = data[512*i:512*i+bufferLength] # Fill the matrix with the buffers
    return buffersMatrix


# Pre-emphasis filter
def preEmphasis(data):
    return np.append(data[0], data[1:] - 0.95*data[:-1])


# Hamming window
def hamming(data):
    return scipy.signal.hamming(len(data))*data


# Mag-spectrum calculation
def magSpectrum(data):
    fft = scipy.fftpack.fft(data)
    return np.abs(fft[:fftLength])


# Convert Hertz to mel
def convertToMel(f):
    return 1127 * np.log(1+(f/700))


# Convert mel to Hertz
def convertToHz(m):
    return 700*(np.exp(m/1127)-1)    


# Return the filters matrix
def getFilters():
    melPoints = np.linspace(0, convertToMel(nyquistF), filtersNumber + 2) # Find the X-axis points of the filters (in mel)
    hzPoints = convertToHz(melPoints) # Convert the X-axis points to Hz
    frequencyBins = hzPoints * fftLength / nyquistF # Convert the Hertz points into frequency bins
    filters = np.zeros([filtersNumber, fftLength]) # Create an empty matrix which will contain the filters
    for i in range(filtersNumber):
        # Convert the frequency bins to integers
        leftBin = int(math.floor(frequencyBins[i]))
        topBin = int(round(frequencyBins[i+1]))
        rightBin = int(math.ceil(frequencyBins[i+2]))
        for j in range(leftBin, topBin):
            filters[i, j] = (j - leftBin) / (topBin - leftBin) # Linear interpolation between the left bin and the top bin
        for j in range(topBin, rightBin):
            filters[i, j] = (rightBin - j) / (rightBin - topBin) # Linear interpolation between the top bin and the right bin
    return filters


# Plot the filters from 0 to a stop frequency
def plotFilters(filters, stopFrequency):
    numberOfPoints = round(stopFrequency * fftLength / nyquistF) # The number of points to plot
    abscissa = np.linspace(0, (numberOfPoints-1) * nyquistF / fftLength, numberOfPoints) # Create the abscissa vector
    filters = np.delete(filters, np.s_[math.floor(numberOfPoints):fftLength], axis=1) # Delete the filters points after the stop frequency
    for i in range(filters.shape[0]):
        plt.pyplot.plot(abscissa, filters[i], marker='o') # Plot each filter
    plt.pyplot.xlabel('Frequency (Hz)')
    plt.pyplot.ylabel('Amplitude')
    plt.pyplot.title('26 triangular MFCC filters, 22050Hz signal, window size 1024')


# Calculate the MFCC
def getMFCC(fileName):
    buffersMatrix = getBuffersMatrix(fileName)
    buffersNumber = buffersMatrix.shape[0]
    mfccMatrix = np.empty([buffersNumber, fftLength]) # Create an empty matrix which will contain the MFCC
    for i in range(buffersNumber):
        mfccMatrix[i] = magSpectrum(hamming(preEmphasis(buffersMatrix[i]))) # First, calculate the mag-specrum
    return scipy.fftpack.dct(np.log10(np.matmul(mfccMatrix, getFilters().T))) # Then, calculate the MFCC and return the MFCC matrix


# Computes the mean and standard deviation of a matrix M along the axis 0
def meanAndStd(M): 
    return np.concatenate([np.mean(M, axis=0), np.std(M, axis=0)])


# Output the data to an ARFF file
def createArffFile():
    with open("perceptualFeatures.arff", "w") as perceptualFeatures:
        perceptualFeatures.write('''@RELATION music_speech
@ATTRIBUTE MFCC_0 NUMERIC
@ATTRIBUTE MFCC_1 NUMERIC
@ATTRIBUTE MFCC_2 NUMERIC
@ATTRIBUTE MFCC_3 NUMERIC
@ATTRIBUTE MFCC_4 NUMERIC
@ATTRIBUTE MFCC_5 NUMERIC
@ATTRIBUTE MFCC_6 NUMERIC
@ATTRIBUTE MFCC_7 NUMERIC
@ATTRIBUTE MFCC_8 NUMERIC
@ATTRIBUTE MFCC_9 NUMERIC
@ATTRIBUTE MFCC_10 NUMERIC
@ATTRIBUTE MFCC_11 NUMERIC
@ATTRIBUTE MFCC_12 NUMERIC
@ATTRIBUTE MFCC_13 NUMERIC
@ATTRIBUTE MFCC_14 NUMERIC
@ATTRIBUTE MFCC_15 NUMERIC
@ATTRIBUTE MFCC_16 NUMERIC
@ATTRIBUTE MFCC_17 NUMERIC
@ATTRIBUTE MFCC_18 NUMERIC
@ATTRIBUTE MFCC_19 NUMERIC
@ATTRIBUTE MFCC_20 NUMERIC
@ATTRIBUTE MFCC_21 NUMERIC
@ATTRIBUTE MFCC_22 NUMERIC
@ATTRIBUTE MFCC_23 NUMERIC
@ATTRIBUTE MFCC_24 NUMERIC
@ATTRIBUTE MFCC_25 NUMERIC
@ATTRIBUTE MFCC_26 NUMERIC
@ATTRIBUTE MFCC_27 NUMERIC
@ATTRIBUTE MFCC_28 NUMERIC
@ATTRIBUTE MFCC_29 NUMERIC
@ATTRIBUTE MFCC_30 NUMERIC
@ATTRIBUTE MFCC_31 NUMERIC
@ATTRIBUTE MFCC_32 NUMERIC
@ATTRIBUTE MFCC_33 NUMERIC
@ATTRIBUTE MFCC_34 NUMERIC
@ATTRIBUTE MFCC_35 NUMERIC
@ATTRIBUTE MFCC_36 NUMERIC
@ATTRIBUTE MFCC_37 NUMERIC
@ATTRIBUTE MFCC_38 NUMERIC
@ATTRIBUTE MFCC_39 NUMERIC
@ATTRIBUTE MFCC_40 NUMERIC
@ATTRIBUTE MFCC_41 NUMERIC
@ATTRIBUTE MFCC_42 NUMERIC
@ATTRIBUTE MFCC_43 NUMERIC
@ATTRIBUTE MFCC_44 NUMERIC
@ATTRIBUTE MFCC_45 NUMERIC
@ATTRIBUTE MFCC_46 NUMERIC
@ATTRIBUTE MFCC_47 NUMERIC
@ATTRIBUTE MFCC_48 NUMERIC
@ATTRIBUTE MFCC_49 NUMERIC
@ATTRIBUTE MFCC_50 NUMERIC
@ATTRIBUTE MFCC_51 NUMERIC
@ATTRIBUTE class {music,speech}
@DATA\n''') # Write the header of the ARFF file
        for fileName in getFileNames():
            perceptualFeatures.write(",".join(str(x) for x in np.round(meanAndStd(getMFCC(fileName)), 6)) + ",") # Write the features (converted to strings) of the file
            perceptualFeatures.write(fileName.split("_")[0]+"\n") # Add the label of the file


plotFilters(getFilters(), 300)
plotFilters(getFilters(), nyquistF)
createArffFile()
