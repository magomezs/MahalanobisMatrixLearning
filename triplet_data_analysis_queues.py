import caffe
import numpy as np
import os
import io
import sys

class TripletDataAnalysisQueues(caffe.Layer):
	def setup(self, bottom, top):
		
		#Checking inputs and outputs
		params = eval(self.param_str)
       		self.snapshot = params["snapshot_step"]
		self.queue_size = params["queue_size"]

		if len(bottom)!=3:
			raise Exception('must have exactly three inputs: 3 descriptors')
		if len(top)!=1:
			raise Exception('one output: MM')
		self.iteration =0
                self.Scola = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)
		self.Dcola = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)
		self.Svar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)
		self.Dvar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)
                

	

		if len(bottom)!=3:
			raise Exception('must have exactly three inputs: 2 features and 1 label')
		if len(top)!=1:
			raise Exception('one output: Mahalanobis Matrix')
			
		#Parameters reading
		params = eval(self.param_str)
       		self.snapshot = params["snapshot_step"]
		self.queue_size = params["queue_size"]
		
		self.iteration =0
                self.Squeue = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)				# Similarity queue
		self.Dqueue = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)				# Dissimilarity queue
		self.Svar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Similarity variations
		self.Dvar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Dissimilarity variations
                self.Y= np.zeros((bottom[0].num, 1), dtype=np.float32)							# Labels array

	
		
		
		
		
		
		

	def reshape(self, bottom, top):
		#check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception('Inputs must have the same dimension')
		if bottom[0].count != bottom[2].count:
			raise Exception('Inputs must have the same dimension')

                top[0].reshape(bottom[0].channels, bottom[0].channels)

		#differnce has shape of inputs
		self.diff_p = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)    #Descripstors difference has descriptors dimensions
		self.diff_n = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32) 
                self.Smean = np.zeros((bottom[0].channels), dtype=np.float32)
		self.Dmean = np.zeros((bottom[0].channels), dtype=np.float32)
		self.S = np.identity((bottom[0].channels), dtype=np.float32)
		#print self.S
		self.D = np.identity((bottom[0].channels), dtype=np.float32)
		self.M = np.identity((bottom[0].channels), dtype=np.float32)
		self.col= np.zeros((bottom[0].channels, 1), dtype=np.float32)
 		self.row= np.zeros((1, bottom[0].channels), dtype=np.float32)
		 	
       		
	def forward(self, bottom, top):
		self.iteration=self.iteration+1
		#print self.iteration
        	self.diff_p = np.absolute(bottom[0].data[...] - bottom[1].data[...])
        	self.diff_n = np.absolute(bottom[0].data[...] - bottom[2].data[...])
		#print self.diff
		#print bottom[0].data[...]
		#print bottom[1].data[...]

		#Move piles
		for i in range(bottom[0].num):
			for a in range(self.queue_size-1):
				self.Scola[a,:]=self.Scola[a+1,:]
			self.Scola[self.queue_size-1,:] = self.diff_p[i,:] 
			#self.Scounter=self.Scounter+1;
			for a in range(self.queue_size-1):
				self.Dcola[a,:] = self.Dcola[a+1,:]
			self.Dcola[self.queue_size-1,:] =self.diff_n[i,:] 
			#self.Dcounter=self.Dcounter+1;
		
		self.Smean = np.sum(self.Scola, axis=0)/float(self.queue_size)
	
	
		#New covariance matrices
		#if (self.iteration%10000)==0 :
		for k in range(self.queue_size):	
			#for e in range(bottom[0].channels)
			self.row[0,:] = (self.Scola[k,:]-self.Smean[:])
			self.col[:,0] = (self.Scola[k,:]-self.Smean[:]).transpose()
			self.Svar[k,:,:] = self.col * self.row #(self.Scola[k,:]-self.Smean[:]).transpose() * (self.Scola[k,:]-self.Smean[:])
		self.S = np.sum(self.Svar, axis=0)/float(self.queue_size)


		#if self.Dcounter>=1000 :
		#mean vector
		self.Dmean = np.sum(self.Dcola, axis=0)/float(self.queue_size)

		#New covariance matrices
		for k in range(self.queue_size):
			self.row[0,:] = (self.Dcola[k,:]-self.Dmean[:])
			self.col[:,0] = (self.Dcola[k,:]-self.Dmean[:]).transpose()
			self.Dvar[k,:,:] = self.col * self.row # (self.Dcola[k,:]-self.Dmean[:]).transpose() * (self.Dcola[k,:]-self.Dmean[:])
	
		self.D = np.sum(self.Dvar, axis=0)/float(self.queue_size)


		#Mahalanobis matrix
	        #if self.Scounter>=1000:
			#if self.Dcounter>=1000:
		self.M=np.absolute(np.linalg.inv(self.S+np.identity(self.S.shape[0])*0.01) - np.linalg.inv(self.D+np.identity(self.D.shape[0])*0.01))
		#print self.M

		#SAVE MM
		top[0].data[...]=self.M
                
                if (self.iteration%self.snapshot)==0 :
			#imprimir matrix en txt
			filename = ("../WEIGHTS/MM_%i.txt" % (self.iteration))
			np.savetxt(filename, self.M)


	def backward(self, bottom, propagate_down, top):
		pass

	
	
	
	...................................................................................................................


class DataAnalysisQueues(caffe.Layer):
	def setup(self, bottom, top):
		#Checking inputs and outputs
		if len(bottom)!=3:
			raise Exception('must have exactly three inputs: 2 features and 1 label')
		if len(top)!=1:
			raise Exception('one output: Mahalanobis Matrix')
			
		#Parameters reading
		params = eval(self.param_str)
       		self.snapshot = params["snapshot_step"]
		self.queue_size = params["queue_size"]
		
		self.iteration =0
                self.Squeue = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)				# Similarity queue
		self.Dqueue = np.zeros((self.queue_size, bottom[0].channels), dtype=np.float32)				# Dissimilarity queue
		self.Svar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Similarity variations
		self.Dvar = np.zeros((self.queue_size, bottom[0].channels, bottom[0].channels), dtype=np.float32)	# Dissimilarity variations
                self.Y= np.zeros((bottom[0].num, 1), dtype=np.float32)							# Labels array


	def reshape(self, bottom, top):
		#check input dimensions match
		if bottom[0].count != bottom[1].count:
			raise Exception('Inputs must have the same dimension')
                #output has Mahalanobis Matrix dimensions
                top[0].reshape(bottom[0].channels, bottom[0].channels)

		self.diff = np.zeros((bottom[0].num, bottom[0].channels), dtype=np.float32)     # Features difference has descriptors dimensions
                self.Smean = np.zeros((bottom[0].channels), dtype=np.float32)			# Similarity expected values array
		self.Dmean = np.zeros((bottom[0].channels), dtype=np.float32)			# Dissimilarity expedted values array
		self.S = np.identity((bottom[0].channels), dtype=np.float32)			# Similarity covariance matrix
		self.D = np.identity((bottom[0].channels), dtype=np.float32)			# Dissimilarity covariance matrix
		self.M = np.identity((bottom[0].channels), dtype=np.float32)			# Mahalanobis matrix
		self.col= np.zeros((bottom[0].channels, 1), dtype=np.float32)                   
 		self.row= np.zeros((1, bottom[0].channels), dtype=np.float32)
		 	
       		
	def forward(self, bottom, top):
		self.iteration=self.iteration+1
		self.Y[:,0] = bottom[2].data[...]
        	self.diff = bottom[0].data[...] - bottom[1].data[...]
	
		#Move FIFO queues
		for i in range(bottom[0].num):
			if self.Y[i] == 1.0 :
				for a in range(self.queue_size-1):
					self.Squeue[a,:]=self.Squeue[a+1,:]
				self.Squeue[self.queue_size-1,:] = self.diff[i,:] 	
			else:
				for a in range(self.queue_size-1):
					self.Dqueue[a,:] = self.Dqueue[a+1,:]
				self.Dqueue[self.queue_size-1,:] =self.diff[i,:] 
				
		# Similarity covariance matrix computation
		self.Smean = np.sum(self.Squeue, axis=0)/float(self.queue_size)
		for k in range(self.queue_size):
			self.row[0,:] = (self.Squeue[k,:]-self.Smean[:])
			self.col[:,0] = (self.Squeue[k,:]-self.Smean[:]).transpose()
			self.Svar[k,:,:] = self.col * self.row
		self.S = np.sum(self.Svar, axis=0)/float(self.queue_size)

		# Dissimilarity covariance matrix computation
		self.Dmean = np.sum(self.Dqueue, axis=0)/float(self.queue_size)
		for k in range(self.queue_size):
			self.row[0,:] = (self.Dqueue[k,:]-self.Dmean[:])
			self.col[:,0] = (self.Dqueue[k,:]-self.Dmean[:]).transpose()
			self.Dvar[k,:,:] = self.col * self.row
		self.D = np.sum(self.Dvar, axis=0)/float(self.queue_size)

		#Mahalanobis matrix computation
		self.M=np.linalg.inv(self.S+np.identity(self.S.shape[0])*0.01) - np.linalg.inv(self.D+np.identity(self.D.shape[0])*0.01)
		top[0].data[...]=self.M
		
		#SAVE MM
                if (self.iteration%self.snapshot)==0 :
			filename = ("../WEIGHTS/MM_%i.txt" % (self.iteration))
			np.savetxt(filename, self.M)
             

	
