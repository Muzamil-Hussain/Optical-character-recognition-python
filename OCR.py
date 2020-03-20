# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:16:47 2019

@author: Muzamil
"""

from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import pickle

dataSetRows = 62
dataSetCols = 30
dataSetSize = 1860
dataSet = np.zeros((dataSetSize,32,32))
labels = np.chararray((dataSetSize))


#function responsible for returning
#labels for specific index
def labelNames(index):
    tempLabels = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
                  'n','o','p','q','r','s','t','u','v','w','x','y','z',
                  'A','B','C','D','E','F','G','H','I','J','K','L','M',
                  'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                  '1','2','3','4','5','6','7','8','9','0']

    return tempLabels[index]


#function responsible for loading
#images from the training data set
#and feeding it to the dataSet array
def loadDataSet():
    labelSize = 0
    for i in range (0,dataSetRows):
        for j in range (dataSetCols):
            labels[labelSize] = labelNames(i)
            dataSet[labelSize] = cv2.imread('trainingData\\'+str(i+1)+'\\'+str(j+1)+'.png',0)
            labelSize+=1


def resizeDataSet():
    for i in range (0,dataSetRows):
        for j in range (dataSetCols):
            img =  cv2.imread('trainingData\\'+str(i+1)+'\\'+str(j+1)+'.png',0)
            img = cv2.resize(img,(32,32))
            img = cv2.imwrite ('trainingData\\'+str(i+1)+'\\'+str(j+1)+'.png',img)



#function responsible for generating contours first
#by words then for every word, it generates letters
# and prepares final array#
def getContours(img, img2, flag, word_no=-1):
    lineContours = []
    res = []
    if flag == True:
        sumOfRows = np.sum(img, axis=1)
        # loop the summed values
        startindex = 0
        lines = []
        compVal = True
        for i, val in enumerate(sumOfRows):
            # logical test to detect change between 0 and > 0
            testVal = (val > 0)
            if testVal == compVal:
                    # when the value changed to a 0, the previous rows
                    # contained contours, so add start/end index to list
                    if val == 0:
                        lines.append((startindex,i))
                    # update startindex, invert logical test
                        startindex = i+1
                    compVal = not compVal

        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        contours, hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for j,cnt in enumerate(contours):
            (x,y,w,h) = cv2.boundingRect(cnt)
            for i,line in enumerate(lines):
                if y >= line[0] and y <= line[1]:
                    res.append((line[0], x, y, x + w, y + h))
                    lineContours.append([line[0],x,j])
                    break

        # sort list on line number,  x value and contour index
        contours_sorted = sorted(lineContours)
        res = sorted(res)
        words_arr = [None] * len(res)
        for i, cnt in enumerate(contours_sorted):
            #cv2.rectangle(img, (res[i][1],res[i][2]), (res[i][3],res[i][4]), color=(255,0,0), thickness=5)
            #if i < 20:
            word = img2[res[i][2]:res[i][4],res[i][1]:res[i][3]]
            word2 = word.copy()
            word = cv2.Canny(word,100,200)
            #if i <= 1:
            words_arr[i] = getContours(word, word2, False, i)

        return words_arr
    else:
        #background = np.zeros((128,128))
        #for i in range(128):
        #    for j in range(128):
        #        background[i][j] = 255
        scale_percent = 220
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)

        #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        kernel = np.zeros((3, 3), np.uint8)
        kernel[1][0] = 1
        kernel[1][1] = 1
        kernel[1][2] = 1
        #print(kernel)
        #img = cv2.dilate(img, kernel, iterations=1)
        #cv2.imshow(str(word_no),img)
        contours, hier = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for j,cnt in enumerate(contours):
            (x,y,w,h) = cv2.boundingRect(cnt)
            area = w*h
            if (area > 166):
                res.append((x, y, x + w, y + h))
                lineContours.append([0,x,j])

        contours_sorted = sorted(lineContours)
        res = sorted(res)
        letter = np.zeros((len(res),32,32))
        for i, cnt in enumerate(contours_sorted):
            alphabet = img2[res[i][1]:res[i][3],res[i][0]:res[i][2]]
            #scale_percent = 300 # percent of original size
            #width = int(alphabet.shape[1] * scale_percent / 100)
            #height = int(alphabet.shape[0] * scale_percent / 100)
            #print((height,width))
            #alphabet = cv2.resize(alphabet,(width,height), interpolation = cv2.INTER_AREA)
            alphabet = cv2.resize(alphabet,(32,32), interpolation = cv2.INTER_AREA)
            letter[i] = alphabet
            #cv2.imwrite('letter'+str(word_no)+str(i)+'.png',alphabet)
            #cv2.imshow('letter_B'+str(word_no)+str(i),alphabet)
        return  letter





# Function which loads images from images folder and prepares dataSet from scratch
#loadDataSet()


#command to save dataSet in the disk
#np.save('dataSet.npy',dataSet)

#command to load dataSet from the disk
#dataSet=np.load('dataSet.npy',mmap_mode='r+')

#converting dataSet into type unsigned integer
#dataSet = dataSet.astype(np.uint8)


# setting cell_size, block_size and no of bins to process HOG Descriptor
cell_size = (8,8)
block_size= (2,2)
nbins = 9


#initializing HOGdescriptor object
hogDesc = cv2.HOGDescriptor(_winSize=(32,32),
                            _blockSize=(block_size[1] * cell_size[1],block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)


#getting descriptor size
descSize = hogDesc.getDescriptorSize()

#initializing descriptor Array of size dataSet(62992)xdescSize(8100)
#descArray = np.zeros((len(dataSet),descSize))


#calculating descriptors for each image and pushing it into 1860*descriptor size array
#for i in range(0,dataSetSize):
#    print(i)
#    descriptor = hogDesc.compute(dataSet[i])
#    for j in range(0,descSize):
#        descArray[i][j] = descriptor[j]


# Commands to save desc Array and labels array in the disk
#np.save('descArray.npy',descArray)
#np.save('labels.npy',labels)

# Commands to load desc Array and labels array from the disk
#descArray = np.load('descArray.npy')
#labels = np.load('labels.npy')

#preparing training model
#Classifier = RandomForestClassifier(n_estimators = 600, max_depth=None)
#Classifier.fit(descArray,labels)

# save the model to disk
filename = 'trained_model.sav'
#pickle.dump(Classifier, open(filename, 'wb'))

# load the model from disk
Classifier = pickle.load(open(filename, 'rb'))



img = cv2.imread('texttotest.png',0)
img2 = img.copy()
h,w = img.shape[:2]
img = cv2.Canny(img,100,200)
img_arr = getContours(img, img2, True)


text = ""
for i in range(len(img_arr)):
    img_arr[i] = img_arr[i].astype(np.uint8)
    for j in range(len(img_arr[i])):
        testDescArr = np.zeros((1,descSize))
        testDescriptor = hogDesc.compute(img_arr[i][j])
        for k in range (0,descSize):
            testDescArr[0][k] = testDescriptor[k]
        prediction = Classifier.predict(testDescArr)
        #print(chr(prediction[0][0]))
        text+=chr(prediction[0][0])
    text+=' '

print(text)
