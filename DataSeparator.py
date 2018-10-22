"""
this file to separate the training data into their corresponding labels 
separate_training()
the final output should be something like that
	-/data
	  -/2ch
	  -/3ch
	  -/4ch
	  -/basal
	  -/mid
	  -/apical
the logic is based on the convention for image naming 

PatientID_Position_SliceID_TimeFrame_augmentationNumber_[mask].tif

second option 

separate data execlusively

 -/data_execlusive
	-/test1/ images of different slices with the same time frame and patientID and position
	-/test2 / ----
	and so on

	number of files = number of time frames * number of patients

"""

import os
import numpy as np
import cv2
from Image import *

exclusive = True
IMAGE_SIZE_MOBILENET = 224
IMAGE_SIZE_INCEPTIONV3 = 299
current_path=os.getcwd()

#TODO
#make it as argument in the cmd
out_path=os.path.join(current_path,"exclusive_test_2")

#TODO 
# add this as an argument in the cmd
dataPath = "/home/minaessam/Documents/original_test/"

def create_subfolders(foldersPaths):

	baseName=os.path.dirname(foldersPaths[0])
	if not os.path.exists(baseName):
		os.mkdir(baseName)
		print("created : "+baseName)

	for subfolder in foldersPaths:
		
		os.mkdir(subfolder)
		print("created : "+subfolder)

def get_files(directory, files = True):
    """
    returns the full names of files or directories inside a given path
    """
    out = [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory,f)) == files ]
    return out


def is_valid_name(fields):
	#fields = imageName.split('_')
	if(len(fields) < 4 or  len(fields) > 8 ):
		return False
	else: 
		return True


def save_images(imagesPath, outPath = None, extension= 'jpg'):
	"""
	Saves batch of images the given out path after doing the preprocessing which could be 
	1- subtract mean
	2- normalize
	3- both of them

	
	"""
	
	#imagesNames = os.listdir(imagesPath)
	Image.set_path(dataPath)
	if outPath == None :
		image = Image( size= IMAGE_SIZE_MOBILENET, outputBasePath = out_path)
	else :
		if( not os.path.exists(outPath)):
			os.mkdir(outPath)
			print("folder created : "+outPath)
		else : 
			#TODO 
			#handle this properly if the images already exists 
			#you may remove them
			pass
		image = Image( size= IMAGE_SIZE_MOBILENET)
	
	for imageName in imagesPath:
		if(is_valid_name(imageName.split('_'))):
			image.read_image(imageName)
			if outPath != None:
				image.save(outputPath=outPath, extension = extension, normalize_mean = False, print_info = False)
			else:
				image.save(extension = extension, normalize_mean = False, print_info = False)
		
		
def separate_exclusive():
	"""
	separate data execlusively

	-/data_execlusive
		-/test1/ images of different slices with the same time frame and patientID and position
		-/test2 / ----
		and so on

		number of files = number of time frames * number of patients
	"""
	#check if the out directory exits
	if not os.path.exists(out_path):
		os.mkdir(out_path)
		print("created : "+out_path)
		
	#make dict with keys time frames and values [images]
	timeFrames = {}
	#set the dataPath for images to be read
	Image.set_path(dataPath)
	image = Image(size = IMAGE_SIZE_MOBILENET)
	for imageName in os.listdir(dataPath):
		#print(imageName)
		#check for images only
		if not (".tif" in imageName.lower() or ".png" in imageName.lower() or ".jpeg"  in imageName.lower() or ".jpg" in imageName.lower()):
			continue
		image.read_image(imageName)
		#don't include mask images
		if image.mask :
			continue
		#the key stored in the dict for image
		key = image.patientID+image.position+image.timeFrame
		if key in timeFrames:
			timeFrames[key].append(image.imageName)
		else:
			timeFrames[key] = [image.imageName]
	#loop over slices and create the folders
	#folder in the above mentioned formats
	#baseSavePath = os.path.join(out_path,"test")
	for i,key in enumerate(timeFrames.keys()):
		#check if incomplete classes 
		if len(timeFrames[key]) != 6:
			print(key, len(timeFrames[key]))
			continue
		savePath = os.path.join(out_path,"test"+str(i))

		save_images(timeFrames[key], outPath= savePath)
			
	



def separate_training():
	"""
	this file to separate the training data into their corresponding labels 
	separate_training()
	the final output should be something like that
		-/data
		-/2ch
		-/3ch
		-/4ch
		-/basal
		-/mid
		-/apical
	the logic is based on the convention for image naming 

	PatientID_Position_SliceID_TimeFrame_augmentationNumber_[mask].tif
	"""

	classes_path=['2ch','3ch','4ch','basal','mid','apical']
	for i in range(len(classes_path)):
		
		classes_path[i]=os.path.join(out_path,classes_path[i])

	if(os.path.exists(out_path)):
		
		for i in range(len(classes_path)):
			
			if(not os.path.exists(classes_path[i])):
				
				print("directory : "+classes_path[i]+"  doesn't exist..creating one")
				#TODO 
				#handle this one gracefully by creating the directory and removing data to 
				#it, and asking the user to repeate that for all missing ones or do all the
				#subfolders
	else:
		print("creating data paths")
		create_subfolders(classes_path)

	save_images(os.listdir(dataPath))


	
if __name__ == "__main__":

	if(not exclusive):
		separate_training()
	else :

		separate_exclusive()


