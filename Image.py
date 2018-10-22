
import numpy as np
import os
import cv2
from shutil import copyfile
class Image:
	"""
	class for image parsing based on their path name
	used to
	1- parse the image path and extract info 
	2- resize the image to the specified size
	3- save the resulting image in the proper place
	4- makes proper preprocessing if specified mean , normalization
	@member fields
	patientID
	position
	sliceID
	timeFrame
	mask  boolean

	example name : 2005_Rest_1_00_0.tif
		       2005_Rest_1_00_0_mask.tif

	"""
	
	__sliceType = { '1' : '2ch'    ,
		        '2' : '3ch'   ,
		        '3' : '4ch'  ,
		        '4' : 'basal' ,
		        '5' : 'mid'   ,
		        '6' : 'apical' }

	__imageOrder = { 'HT' 	   : 0,
			 'LT'	   : 1,
			 'anatomy' : 2 }

	imagesPath = ""





	def __init__(self, size, outputBasePath =None, interpolation = 'area'):
		
		
		
		self.outputBasePath = outputBasePath
		self.size = size
		
		if (interpolation == 'area' ):
			self.interpolation = cv2.INTER_AREA
		elif (interpolation == 'linear' ):
			self.interpolation = cv2.INTER_LINEAR



	def set_path( path):
		Image.imagesPath = path
	
	def resize(self):
		""" 
		if the image is square resize directly 
		if the aspect ratio not 1 i.e not square the ....		
		"""

		if ( self.image.shape[0] == self.size ):
			
			return
		elif ( self.image.shape[0] == self.image.shape[1] ):
	 

			self.image=cv2.resize(self.image, (self.size,self.size), self.interpolation)

		else :
			print("not equal sizes")

	def read_image(self, imageName):
		
		if Image.imagesPath == "" :
			assert True , "base location for images not provided use  set_path(path) "
		self.imageName = imageName

		fields = self.imageName.split('_')
		if 'mask' in self.imageName.lower():
			self.mask=True
		else: 
			self.mask = False

	
		self.patientID = fields[0]
		self.position = fields[1]
		self.sliceID = fields[2]
		self.timeFrame = fields[3]

		
		self.image=cv2.imread(os.path.join(Image.imagesPath, self.imageName))
		#print(self.mask)
		self.resize()


	def save(self,outputPath=None, savemask = False, extension = 'tif',normalize_mean = False, normalize_range = False, print_info = False):

		"""
		save the current image to the correct location or any desired location in case 
		of not providing output paths	
		it also saves the image with specified format
		PNG , TIF , JPG  (case insensitive)

		outputPath : if given the image will be saved there otherwise outputBasePath should be set in advance 
		savemask : save the image if it's mask
		extension : the extension to be added to the image
		normalize_mean : subtract the mean before saving
		normalize_range : normalize before saving
		print_info : print image info before saving like mean and std (for debuging purposes)
		"""		
		
		
		newImageName=self.imageName
		fields = newImageName.split('.')
		fields[-1] = extension
		newImageName = ''
		#get all the name without the extension
		for i in range(len(fields)-1) :

			newImageName += fields[i]
		#add the extension
		newImageName += '.'+extension
		
		if (normalize_mean == True):
			mean = np.array([np.mean(self.image[:,:,0]), np.mean(self.image[:,:,1]), np.mean(self.image[:,:,2])])
			self.image = (self.image - mean)
		if normalize_range == True :
			self.image= (self.image.astype(dtype = np.float64)-np.amin(self.image))/(np.amax(self.image)-np.amin(self.image)+1e-7) 
		if(print_info == True) :
			print("type %s "%self.image.dtype)
			print("mean : %2.3f"%np.mean(self.image))
			print("std : %2.3f"% np.std(self.image))
		
		if(not savemask ):
			if( self.mask):
					#print("returning")
					return
			
		
		
			
		if(outputPath == None ):


		
			if (self.outputBasePath == None ):
				assert False , "output paths not provided or any customized path "
			else :
				outPath = os.path.join(self.outputBasePath ,Image. __sliceType[self.sliceID])
				finalOutPath=os.path.join(outPath, newImageName)
				if (extension == 'tif'):

					copyfile(os.path.join(Image.imagesPath, self.imageName), finalOutPath)
				else :
					 cv2.imwrite(finalOutPath, self.image )
				
		else :
			finalOutPath=os.path.join(outputPath, newImageName)
			if (extension == 'tif'):

				copyfile(os.path.join(Image.imagesPath, self.imageName), finalOutPath)
			else :
				cv2.imwrite(finalOutPath, self.image )
							
		
			
			
		
		
