import caffe
import numpy as np
import os
import io
import sys

class DataAnalysisWeightedAverages(caffe.Layer):
	def setup(self, bottom, top):
		if len(bottom)!=3:
			raise Exception('must have exactly three inputs: 2 descriptors and 1 label')
		if len(top)!=1:
			raise Exception('one output: MM')
		self.iteration = 0
                self.Y= np.zeros((bottom[0].num, 1), dtype=np.float32)
		params = eval(self.param_str)
		self.beta = params["beta"]
       		self.snapshot = params["snapshot_step"]

		self.S = np.identity((bottom[0].channels), dtype=np.float32)
		self.D = np.identity((bottom[0].channels), dtype=np.float32)


	def reshape(self, bottom, top):
		#check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception('Inputs must have the same dimension')

                top[0].reshape(bottom[0].channels, bottom[0].channels)

		#differnce has shape of inputs
		self.diff = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)    #Descripstors difference has descriptors dimensions
                self.Smean = np.zeros((bottom[0].channels), dtype=np.float32)
		self.Dmean = np.zeros((bottom[0].channels), dtype=np.float32)
		self.Stemp = np.identity((bottom[0].channels), dtype=np.float32)
		self.Dtemp = np.identity((bottom[0].channels), dtype=np.float32)
		self.M = np.identity((bottom[0].channels), dtype=np.float32)
		 	
       		
	def forward(self, bottom, top):
		self.iteration=self.iteration+1
		self.Y[:,0] = bottom[2].data[...]
        	self.diff = bottom[0].data[...] - bottom[1].data[...] 

		Slist=[]
		Dlist=[]

		#Get discriminative sets
		for i in range(bottom[0].num):
			if self.Y[i] == 1.0 :
				Slist.append(self.diff[i,:])
			else:
				Dlist.append(self.diff[i,:])
                
 		#Move set lists to sets arrays
		Ssize=len(Slist)
		Dsize=len(Dlist)
                Sset = np.zeros((Ssize, bottom[0].channels), dtype=np.float32)
		Dset = np.zeros((Dsize, bottom[0].channels), dtype=np.float32)

		for s in range(Ssize):
			Sset[s,:] = Slist[s]
		for d in range(Dsize):
			Dset[d,:] = Dlist[d]


		#variation matrixes
		Svar = np.zeros((Ssize, bottom[0].channels, bottom[0].channels), dtype=np.float32)
		Dvar = np.zeros((Dsize, bottom[0].channels, bottom[0].channels), dtype=np.float32)

		self.Smean = np.sum(Sset, axis=0)/float(Ssize)
	
		for s in range(Ssize):
			Svar[s,:,:]=(Sset[s,:]-self.Smean[:]).transpose() * (Sset[s,:]-self.Smean[:])
		self.Stemp = np.sum(Svar, axis=0)/float(Ssize)



		self.Dmean = np.sum(Dset, axis=0)/float(Dsize)
		
		for d in range(Dsize):
			Dvar[d,:,:]=(Dset[d,:]-self.Dmean[:]).transpose() * (Dset[d,:]-self.Dmean[:])
		self.Dtemp = np.sum(Dvar, axis=0)/float(Dsize)

                #weighted averages
                self.S=self.beta*self.S+(1-self.beta)*self.Stemp
                self.D=self.beta*self.D+(1-self.beta)*self.Dtemp

		#Mahalanobis matrix
		self.M=np.linalg.inv(self.S+np.identity(self.S.shape[0])*0.01) - np.linalg.inv(self.D+np.identity(self.D.shape[0])*0.01)

		#SAVE MM
		top[0].data[...]=self.M
                
                if (self.iteration%self.snapshot)==0 :
			#imprimir matrix en txt
			filename = ("../WEIGHTS/MM_%i.txt" % (self.iteration))
			np.savetxt(filename, self.M)
             


	def backward(self, bottom, propagate_down, top):
		pass
