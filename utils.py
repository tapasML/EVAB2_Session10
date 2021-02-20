"""
This module contains utilities to display sample images.
Can be used for additional functions.
"""
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import dataloader
from dataloader import *
from testing import *
from training import *
import os
import cv2
import os.path as osp


# display some training images in a grid
def displaysampleimage():    
    _dataiter = iter(dataloader.trainloader)
    _images, _labels = _dataiter.next()
    print('shape of images', _images.shape)
    _sample_images = _images[0:4,:,:,:] # first 4 images
   
    # show images
    __imshow__(torchvision.utils.make_grid(_sample_images))
    # print labels
    print(' '.join('%5s' % classes[_labels[j]] for j in range(4)))
    
   
# diaplay an image
def __imshow__(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()     
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    

# display 25 misclassified images in a grid
def displaymisclassified():
    global misclassifiedImages
    misclassifiedImages = misclassifiedImages[0:25]    
    fig = plt.figure(figsize=(10,10))
    plt.title("Misclassified images\n\n")
    plt.axis("off")   
    for index in range(25):
        ax = fig.add_subplot(5, 5, index + 1, xticks=[], yticks=[])       
        image = misclassifiedImages[index]
        image = image / 2 + 0.5
        image = image.cpu().numpy()
        image = np.transpose(image, (1,2,0))
        pred = misclassifiedPredictions[index].cpu().numpy()
        target = misclassifiedTargets[index].cpu().numpy()
        ax.imshow(image.squeeze())    
        ax.set_title(f'pred:{classes[pred]},target={classes[target]}')
        plt.subplots_adjust(wspace=1, hspace=1.5)
    plt.show()
    
# display classwise test accuracy %
def displayaccuracybyclass():
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


# display training accuracy curve
def displayAccuracy():
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    axs.set_title('Train/Validation Accuracy')
    axs.set_xlabel('epoch')
    axs.set_ylabel('% accuracy')
    p1, = plt.plot(testAccuracy, label='Validation')

    plt.legend(handles=[p1], title='Validation Accuracy', \
               bbox_to_anchor=(1.05, 1), loc='upper left')

def displayimagesfrompath(): 
    filenames = []
    gradcam_images = []
    images = []
    titles = []
    for name in os.listdir("results/"):          
        filenames.append('results/'+name)        
   
    for imgname in filenames:
        img = cv2.imread(imgname)        
        gradcam_images.append(img)   
        #print('gram cam added', filenames)
    fig = plt.figure(figsize=(15,15))
    plt.title("Grad CAM for misclassified images\n\n")
    plt.axis("off")   
    for index in range(25):
        ax = fig.add_subplot(5, 5, index + 1, xticks=[], yticks=[])
        image = gradcam_images[index]        
        ax.imshow(image.squeeze())        
    plt.show()
    cleanupresults()

def cleanupresults():   
    for f in os.listdir("results/"):   
        os.remove(os.path.join("results/", f))    

def plotAccuracy():
    global train_accuracies, test_accuracies
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    axs.set_title('Train/Validation Accuracy')
    axs.set_xlabel('epoch')
    axs.set_ylabel('% accuracy')
    
    p1, = plt.plot(train_accuracies,label='Training Accuracy')
    p2, = plt.plot(test_accuracies, label='Validation Accuracy')
  

    plt.legend(handles=[p1, p2], title='Train/ Validation Accuracy', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()