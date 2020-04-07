import caffe
import numpy as np
from numpy import *
import yaml
from multiprocessing import Process, Queue
from caffe._caffe import RawBlobVec
from sklearn import preprocessing
import math


class TripletSelectLayer(caffe.Layer):
    def setup(self, bottom, top):
        """Setup the TripletSelectLayer."""
        #self.triplet = config.BATCH_SIZE/3
        self.input_batch=bottom[0].num
        self.output_batch=self.input_batch/2#32
        #print self.input_batch
        #print self.output_batch
        top[0].reshape(self.output_batch,shape(bottom[0].data)[1])
        top[1].reshape(self.output_batch,shape(bottom[1].data)[1])
        top[2].reshape(self.output_batch,shape(bottom[2].data)[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        #print self.input_batch
        #print self.output_batch

        top_archor = []
        top_positive = []
        top_negative = []
        #self.tripletlist = []
        self.mid_hard=[]
        self.hard=[]
        self.residuals=[]
        aps = []
        ans = []

      
        bottom_anchor = []
        bottom_positive = []
        bottom_negative = []

        for i in range((bottom[0]).num):   
            bottom_anchor.append(bottom[0].data[i])
            bottom_positive.append(bottom[1].data[i]) 
            bottom_negative.append(bottom[2].data[i])

        for i in range(((bottom[0]).num)):
            a = np.array(bottom_anchor[i])
            p = np.array(bottom_positive[i])
            n = np.array(bottom_negative[i])
            a_p = a - p
            a_n = a - n
            ap = np.dot(a_p,a_p)
            an = np.dot(a_n,a_n)   
            aps.append([ap, i])
	    ans.append([an, i])

        #aps = sorted(aps.items(), key = lambda d: d[1], reverse = True)
      
        ans_sorted = sorted(ans, key = lambda d: d[0], reverse = False)  
 
        count=0
 	i=0
	while (i<self.output_batch):
             indx=ans_sorted[i][1]
             i=i+1
             if (aps[indx][0]) < (ans[indx][0]): 
             	self.mid_hard.append(indx)
             else:
                self.hard.append(indx)


      
        #if (len(self.mid_hard)<self.output_batch) :
        #     print "only very-hard negatives"
       	#     self.mid_hard=self.hard

        for j in range(len(self.mid_hard)): 
	     n=self.mid_hard[j]
             self.residuals.append(n)
             top[0].data[j] = bottom[0].data[n]
             top[1].data[j] = bottom[1].data[n]
             top[2].data[j] = bottom[2].data[n]
        i=len(self.hard)-1
        p=len(self.mid_hard)
        while (p < self.output_batch):
             n=self.hard[i]
             top[0].data[p] = bottom[0].data[n]
             top[1].data[p] = bottom[1].data[n]
             top[2].data[p] = bottom[2].data[n]
             i=i-1
             p=p+1
    



    def backward(self, top, propagate_down, bottom):  
        bottom[0].diff[...] = np.zeros(shape(bottom[0].diff[...]))
        bottom[1].diff[...] = np.zeros(shape(bottom[1].diff[...]))
        bottom[2].diff[...] = np.zeros(shape(bottom[2].diff[...]))
        #if (len(self.mid_hard)<self.output_batch) :
        # 		self.mid_hard=self.hard

	# for i in range(len(self.hard)):
	# 	e=self.residuals[i]

	# 	if e in self.mid_hard:
# 		        bottom[0].diff[e] = top[0].diff[i]# 
	# 	        bottom[1].diff[e] = top[1].diff[i]
	# 	        bottom[2].diff[e] = top[2].diff[i]

        for i in range(len(self.residuals)):
		e=self.residuals[i]
                bottom[0].diff[e] = top[0].diff[i]
	        bottom[1].diff[e] = top[1].diff[i]
	        bottom[2].diff[e] = top[2].diff[i]

        #print 'backward-no_re:',bottom[0].diff[0][0]
        #print 'tripletlist:',self.no_mid_hard

        

    def reshape(self, bottom, top):
        pass





