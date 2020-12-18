import tensorflow as tf
#import pandas as pd
import numpy as np
#import time
import matplotlib.pyplot as plt
#import multiprocessing
#from tensorflow import keras
#from tensorflow.keras import layers
#import tensorflow_addons as tfa
import tensorflow_datasets as tfds
#import shutil
#from datetime import datetime
#from tensorflow.keras.callbacks import TensorBoard
#import time
import os
#import io
import cv2

# Reading the image using imread() function 
# OPEN CV reads images in BGR not in RGB!!!!
def print_test_image(path):
    image = cv2.imread(path)[:,:,::-1]
    
    #print shape of the image
    print(image.shape)
    
    # Extracting the height and width of an image 
    h, w = image.shape[:2]
    
    # Displaying the height and width 
    print("Height = {},  Width = {}".format(h, w)) 
    
    #plot with matplotlib!
    image = image/255
    lum_image = image[:, :, 0]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(image)
    ax.set_title('Before')
    #plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(lum_image)
    imgplot.set_clim(0.0, 0.7)
    ax.set_title('After')
    plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    
    plt.show()

def print_test_depth(path):
    depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    #print shape of the image
    print(depth.shape)
        
    # Extracting the height and width of an image 
    h, w = depth.shape[:2]
        
    # Displaying the height and width 
    print("Height = {},  Width = {}".format(h, w)) 
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)#, projection='3d')
    ax1.set_xlabel("Height")
    ax1.set_ylabel("Width")
    ax1.set_title('Depth')
    depth[depth > 4] = 0
    depth *= 50
    plt.imshow(depth)
    plt.colorbar(spacing='proportional')
    
    #ax2 = fig.add_subplot(111, projection='3d')
    #ax2.set_xlabel("Height")
    #ax2.set_ylabel("Width")
    #ax2.set_zlabel("Depth")
    #x =  range(0,511)
    #y =  range(0,511)
    #x,y = np.meshgrid(x, y)
    #depth /= 5
    #ax2.scatter(xs=x,ys=y,zs= -depth[x,y],s=1,cmap='viridis')
    #ax2.plot_wireframe(X=x,Y=y,Z= -depth[x,y])
    
    plt.show()

#Remove _RGB_ and _DEPTH_ from the file names to ensure same naming...
def Prepare_filename(directory):
    for filename in os.listdir(directory):
        #os.rename(os.path.join(directory, filename), os.path.join(directory, newFilename))
        if "_RGB_" in filename:
            newFilename = filename.replace("_RGB_","_")
            os.rename(os.path.join(directory, filename), os.path.join(directory, newFilename))
        elif "_DEPTH_" in filename:
            newFilename = filename.replace("_DEPTH_","_")
            os.rename(os.path.join(directory, filename), os.path.join(directory, newFilename))

#loads RGB and DEPTH images and combines them into one RGBD Dataset. Shape (sample, Height, Width, Channel). Channel: R,G,B,D
def Combine_color_depth(directory):
    
    #get list from all filenames without file extension. Note that RGB and EXR need to have the same name!
    fileNames = [".".join(f.split(".")[:-1]) for f in os.listdir(directory)]
    
    #Hack to remove duplicates from List
    fileNames = list(dict.fromkeys(fileNames))
    
    numberOfFiles = len(fileNames)

    #empty array container
    combined = np.zeros(shape = (numberOfFiles, 512, 512, 4), dtype=float)

    for i in range(0, numberOfFiles):
        currentFile = os.path.join(directory,fileNames[i])
        image = cv2.imread(currentFile + '.png')[:,:,::-1]
        depth = cv2.imread(currentFile + '.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        print("Iteration {} of {}: Image = {},  Depth = {}".format(i, numberOfFiles, image.shape, depth.shape))
        
        image = image/255;
        
        combined[i,:,:,0:3] = image
        combined[i,:,:,3] = depth
    print("Combined = {}".format(combined.shape))
    
    return combined

def Plot_RGBD_Sample(rgbd):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(rgbd[:,:,0:3])
    ax.set_title('RGB')
    #plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    ax = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(rgbd[:,:,3])
    #imgplot.set_clim(0.0, 0.7)
    ax.set_title('Depth')
    #plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    plt.show()

def Plot_Multiple_RGBD(rgbd_samples, count = 6):
    fig = plt.figure()
    samplesToPlot = min(rgbd_samples.shape[0], count)
    for i in range(0, samplesToPlot):
        ax = fig.add_subplot(3, 4, i*2+1)
        imgplot = plt.imshow(rgbd_samples[i,:,:,0:3])
        ax.set_title('RGB')
        #plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
        ax = fig.add_subplot(3, 4, i*2+2)
        imgplot = plt.imshow(rgbd_samples[i,:,:,3])
        #imgplot.set_clim(0.0, 0.7)
        ax.set_title('Depth')
        #plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
    plt.show()

def Create_tensorflow_dataset(data):
    dataset = tf.data.Dataset.from_tensors(data)
    #print(dataset)
    #dataset = dataset.take(3)
    #print(list(dataset.as_numpy_iterator()))


    #path = os.path.join("Datasets","dataset")
    #tf.data.experimental.save(dataset, path)
   # new_dataset = tf.data.experimental.load(path, tf.TensorSpec(shape=(512,512,4), dtype=tf.float32))

    #new_dataset = dataset.take(3)
    #print(list(new_dataset.as_numpy_iterator()))
    #print(new_dataset)
    return dataset


directory = os.path.join("DatasetVolograms","Synthetic_Data_Samples")
#print_test_image('test.png')
#print_test_depth('test.exr')
#Prepare_filename(directory):
rgbd_samples = Combine_color_depth(directory)
Plot_RGBD_Sample(rgbd_samples[24,:,:,:])
Plot_Multiple_RGBD(rgbd_samples)
#Create_tensorflow_dataset(rgbd_samples)