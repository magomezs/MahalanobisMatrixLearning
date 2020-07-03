import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing

class TripletMahaLoss(caffe.Layer):    
    def setup(self, bottom, top):
        #CHECK DIMENSIONS
        assert shape(bottom[0].data) == shape(bottom[1].data)
        assert shape(bottom[0].data) == shape(bottom[2].data)
               
		    #PARAMS
        self.learning_rate=0.0001
        self.a=2
        self.changing_distance_it = 20000
        self.margin = 0.2
       	self.snapshot = 2000
	      self.iteration=0
        self.MM=np.identity((bottom[0].channels), dtype=np.float32) #MAHALANOBIS MATRIX

        #OUTPUT SHAPE
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)


    def reshape(self, bottom, top):
        #VARIABLES DECLARATION
        self.k=bottom[0].num 
        self.x_a=np.zeros((self.k,1), dtype=np.float32)
        self.x_p=np.zeros((self.k,1), dtype=np.float32)
        self.x_n=np.zeros((self.k,1), dtype=np.float32)


    def forward(self, bottom, top):
        #VARIABLES INITIALISATION
	      self.iteration=self.iteration+1
        loss = float(0)
        hard_well_classified=0	
	      soft_well_classified=0
        self.no_residual_list = []
  
        #INPUTS DIFFERENCES
        self.diff_p = bottom[0].data[...] - bottom[1].data[...]
	      self.diff_n = bottom[0].data[...] - bottom[2].data[...]

        if (self.iteration < self.changing_distance_it):      #USE EUCLIDEAN DISTANCE AS CONNECTION FUNCTION
		      for b in range(bottom[0].num):
		        k=bottom[0].channels
		        self.x_a = reshape(bottom[0].data[b], (k, 1))
		        self.x_p = reshape(bottom[1].data[b], (k, 1))
		        self.x_n = reshape(bottom[2].data[b], (k, 1))
            ap=np.matmul(np.transpose(self.x_a-self.x_p), (self.x_a-self.x_p))
            an=np.matmul(np.transpose(self.x_a-self.x_n), (self.x_a-self.x_n))
            dist = (self.margin + ap - an)
		        _loss = max(dist, 0.0)

    		    if _loss == 0 :
	        		hard_well_classified=hard_well_classified+1
		          self.no_residual_list.append(b)
            if (ap<an):
	 		        soft_well_classified=soft_well_classified+1
		        loss += _loss
        else:                                                  #USE MAHALANOBIS DISTANCE AS CONNECTION FUNCTION
		      for b in range(bottom[0].num):
		      k=bottom[0].channels
		      self.x_a = reshape(bottom[0].data[b], (k,1))
		      self.x_p = reshape(bottom[1].data[b], (k,1))
		      self.x_n = reshape(bottom[2].data[b], (k,1))
		      ap=np.matmul(np.transpose(self.x_a-self.x_p), np.matmul(self.MM, (self.x_a-self.x_p)))
		      an=np.matmul(np.transpose(self.x_a-self.x_n), np.matmul(self.MM, (self.x_a-self.x_n)))
		      dist = (self.margin + ap - an)
		      _loss = max(dist, 0.0)

          if _loss == 0 :
			      hard_well_classified=hard_well_classified+1
		        self.no_residual_list.append(b)
		      if (ap<an):
	 		      soft_well_classified=soft_well_classified+1
		      loss += _loss
        
        #OUTPUTS CALCULATION
        loss = (loss/(bottom[0]).num)
        if (self.iteration%self.snapshot)==0 :
	          filename = ("../WEIGHTS/MM_%i.txt" % (self.iteration))
            np.savetxt(filename, self.MM)

        top[0].data[...] = loss
	      top[1].data[...] = float(hard_well_classified)/float(bottom[0].num)
	      top[2].data[...] = float(soft_well_classified)/float(bottom[0].num)
    

    def backward(self, top, propagate_down, bottom):
        if propagate_down[0]:
	        MM_diff=np.zeros((bottom[0].num, bottom[0].channels, bottom[0].channels),dtype=np.float32 )
  	      MM_diff_mean=np.zeros((bottom[0].channels, bottom[0].channels),dtype=np.float32 )          
          for b in range(bottom[0].num):
                if b in self.no_residual_list:
                    bottom[0].diff[b] = np.zeros(shape(bottom[0].data)[1], dtype=np.float32)
                    bottom[1].diff[b] = np.zeros(shape(bottom[0].data)[1], dtype=np.float32)
                    bottom[2].diff[b] = np.zeros(shape(bottom[0].data)[1], dtype=np.float32)
                    MM_diff[b] = np.zeros((bottom[0].channels, bottom[0].channels), dtype=np.float32)
                else:
		                k=bottom[0].channels
		                Ik=np.identity(k, dtype=np.float32)
                    vec_Ik= reshape(np.identity(k, dtype=np.float32), (k*k, 1))
		                dMM_dMM=np.zeros((k*k, k*k), dtype=np.float32)
                    for m in range(k):
		    	            i=(m*k)+m
		    	            for n in range(k):
		    		            j=(n*k)+n
		    		            dMM_dMM[i,j]=1
                    MM_diff[b] =  (self.a/bottom[0].num) * (np.matmul(np.kron(np.transpose(self.x_n-self.x_p), Ik), np.matmul(dMM_dMM, np.kron((self.x_n-self.x_p), Ik))))

                    if(self.iteration < self.changing_distance_it):
		                    bottom[0].diff[b] =  self.a*np.reshape((self.x_n - self.x_p), k)#/((bottom[0]).num))
		                    bottom[1].diff[b] =  self.a*np.reshape((self.x_p - self.x_a), k)#/((bottom[0]).num))
		                    bottom[2].diff[b] =  self.a*np.reshape((self.x_a - self.x_n), k)#/((bottom[0]).num))    
		                else:
	 		                  bottom[0].diff[b] =  (self.a) * np.reshape(np.matmul(Ik, np.matmul(self.MM, self.x_n-self.x_p)) + matmul(np.kron(np.transpose(self.x_n-self.x_p), Ik), np.matmul(np.kron(self.MM, Ik), vec_Ik)), k)
			                  bottom[1].diff[b] =  (self.a) * np.reshape(np.matmul(Ik, np.matmul(self.MM, self.x_p-self.x_a)) + matmul(np.kron(np.transpose(self.x_p-self.x_a), Ik), np.matmul(np.kron(self.MM, Ik), vec_Ik)), k)
			                  bottom[2].diff[b] =  (self.a) * np.reshape(np.matmul(Ik, np.matmul(self.MM, self.x_a-self.x_n)) + matmul(np.kron(np.transpose(self.x_a-self.x_n), Ik), np.matmul(np.kron(self.MM, Ik), vec_Ik)), k)
			   
	          previous_MM=self.MM
            MM_diff_mean=np.sum(MM_diff, axis=0)/bottom[0].num
            self.MM=previous_MM-self.learning_rate*MM_diff_mean
        
       

