import scipy.misc
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv2
from random import randint
import glob
from PIL import Image
import random
def extract_ch(flag,synt,real_image_stack,i): #This function should be recursively ideally
    rMin=[]
    sMin=[]
    rMax=[]
    sMax=[]
    sMean=[]
    rMean=[]
    ImN=[]
    num=randint(0, flag-1)
    stemp=synt[:,:,i]
    rtemp=real_image_stack[num,:,:,i]
    sMean=(np.mean(stemp.flatten()))
    rMean=(np.mean(rtemp.flatten()))
    if((abs(sMean-rMean)<=0.4)):#thresholding mean for retrieving as similar looking images as possible
        print('enter if')
        rMin.append(min(rtemp.flatten()))
        sMin.append(min(stemp.flatten()))
        rMax.append(max(rtemp.flatten()))
        sMax.append(max(stemp.flatten()))  
    return rMin,sMin,rMax,sMax,rMean,sMean
def min_max_allscale(filename2,real_image_stack,flag):
    syn=(scipy.misc.imread(filename2))

    synt=cv2.normalize(syn.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    rMin=[]
    sMin=[]
    rMax=[]
    sMax=[]
    sMean=[]
    rMean=[]
    ImN=[]
    cnt=0
    while(np.shape(rMin)[0]!=3 and np.shape(sMin)[0]!=3 and np.shape(rMax)[0]!=3 and np.shape(sMax)[0]!=3 and cnt<=1993):#Change cnt to the max no. of synthetic images: If not sure set to the largest value possible   
            for i in range(3):

                num=randint(0, flag-1)
                stemp=synt[:,:,i]
                rtemp=real_image_stack[num,:,:,i]
                sMean=(np.mean(stemp.flatten()))
                rMean=(np.mean(rtemp.flatten()))
                if((abs(sMean-rMean)<=0.4)): #adjust threshold to your needs
                    rMin.append(min(rtemp.flatten()))
                    sMin.append(min(stemp.flatten()))
                    rMax.append(max(rtemp.flatten()))
                    sMax.append(max(stemp.flatten()))  
                while(~((abs(sMean-rMean)<=0.4) and cnt<=5)):
                    cnt=cnt+1
                    rMin,sMin,rMax,sMax,rMean,sMean=extract_ch(flag,synt,real_image_stack,i)
                    break
            else:
                return synt
    if(np.shape(rMin)[0]==3 and np.shape(sMin)[0]==3 and np.shape(rMax)[0]==3 and np.shape(sMax)[0]==3):
        for j in range(3):
            #Shift distribution to zero
            temp1=(synt[:,:,j]-sMin[j])
            #scale max of the real-min of the real
            temp2=temp1*((rMax[j]-rMin[j])/(sMax[j]-sMin[j]))
            # Add the minimum value from the real imagery
            temp3=temp2+rMin[j]
            #Converting scale.
            #cvuint8 = cv2.convertScaleAbs(temp3)
            ImN.append(temp3)
        norma=np.moveaxis(ImN, 0, 2)
        return norma

#For testing on individual chips
filesyn='/link/to/syn'#Link to 'A' synthetic chip
syn=(scipy.misc.imread(filesyn))
real_image_stack = np.stack([np.expand_dims(((Image.open(filename).resize((55,35)))),-1) for filename in glob.glob('link/to/real/*.png')],0) 	#Change link to the folder containing 'ALL' the real imagery you have for reference
real_image_stack=np.squeeze(real_image_stack)	#My data had unwanted channels i.e [6322,224,224,3,1] ,comment out incase your data imports in the form [6322,224,224,3]
real_image_stack=cv2.normalize(real_image_stack.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
flag=(np.shape(real_image_stack))[0]
test=min_max_allscale(filesyn,real_image_stack,flag)

plt.figure()
plt.axis('off')
plt.imshow(test)
plt.figure()
plt.axis('off')
plt.imshow(syn)


#For processing a complete dataset in one go
"""
name=0
real_image_stack = np.stack([np.expand_dims(((Image.open(filename).resize((55,35)))),-1) for filename in glob.glob('link/to/real/*.png')],0)       #Change link to the folder containing 'ALL' the real imagery you have for reference
real_image_stack=np.squeeze(real_image_stack)	#My data had unwanted channels i.e [6322,224,224,3,1] ,comment out incase your data imports in the form [6322,224,224,3]
real_image_stack=cv2.normalize(real_image_stack.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
flag=(np.shape(real_image_stack))[0]
print(np.shape(real_image_stack))
for filename in glob.glob('link/to/synthetic/*.PNG'): #Change to synthetic directory
    normali=min_max_allscale(filename,real_image_stack,flag)
    scipy.misc.imsave('/home/local/KHQ/bhavan.vasu/elm/out/july12/chip{}.png'.format(name),  normali)  #Location of the folder you want your preprocessed images to be saved
    name=i+1
print(ff)
"""
