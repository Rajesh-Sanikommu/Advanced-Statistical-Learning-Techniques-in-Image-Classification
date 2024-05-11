#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io


# In[2]:


testdata = scipy.io.loadmat(r"C:\Users\DELL\Downloads\test_data.mat")
traindata = scipy.io.loadmat(r"C:\Users\DELL\Downloads\train_data.mat")


# In[3]:


testdata['data']


# In[4]:


l=[]
for i in traindata['data']:
    i=i.flatten()
    l.append(i)

traindata['data']=np.array(l)
    
l=[]
for i in testdata['data']:
    i=i.flatten()
    l.append(i)
testdata['data']=np.array(l)
print(len(traindata['data']),len(testdata['data']))


# In[5]:


def cal_skewvec(data):
    skewvec= []
    for i in data:
        mean= np.mean(i)
        std_dev= np.std(i)
        cubestd= pow(std_dev,3)
        n= len(i)
        a=0
        for j in i:
            a+=pow(j-mean,3)
        a/= n
        a/= cubestd
        skewvec.append(a)
    return skewvec


# In[6]:


def BDratio(array,threshold):
    BDratio=[]
    for a in array:
        n=len(a)
        count=0
        for i in a:
            if(i>threshold):
                 count+=1
        BDratio.append(count/(n-count))
        
    return BDratio


# In[7]:


skewvec_train= cal_skewvec(traindata['data'])
BDratio_train= BDratio(traindata['data'],150)
skewvec_test= cal_skewvec(testdata['data'])
BDratio_test= BDratio(testdata['data'],150)
print(len(skewvec_train))
print(len(BDratio_train))


# In[8]:


mean_skewnormal= np.mean(skewvec_train)
std_skewnormal= np.std(skewvec_train)
mean_BDnormal= np.mean(BDratio_train)
std_BDnormal= np.std(BDratio_train)


# In[9]:


skewvec_train= (skewvec_train-mean_skewnormal)/std_skewnormal
BDratio_train= (BDratio_train-mean_BDnormal)/std_BDnormal
skewvec_test= (skewvec_test-mean_skewnormal)/std_skewnormal
BDratio_test= (BDratio_test-mean_BDnormal)/std_BDnormal


# In[10]:


traindata_new= np.stack([skewvec_train,BDratio_train],axis=1)
testdata_new= np.stack([skewvec_test,BDratio_test],axis=1)
print(len(traindata_new),len(testdata_new))


# In[11]:


classofImage_3=[]
classofImage_7=[]
for i in range(len(traindata['label'][0])):
    if traindata['label'][0][i] == 3:
        classofImage_3.append(traindata_new[i])
    else :
        classofImage_7.append(traindata_new[i])

classofImage_3_Array = np.array(classofImage_3)
classofImage_3_Mean = np.mean(classofImage_3_Array, axis=0)
classofImage_3_Covariance = np.cov(classofImage_3_Array[:, 0], classofImage_3_Array[:, 1])

classofImage_7_Array = np.array(classofImage_7)
classofImage_7_Mean = np.mean(classofImage_7_Array, axis=0)
classofImage_7_Covariance = np.cov(classofImage_7_Array[:, 0], classofImage_7_Array[:, 1])

print("meanof3",classofImage_3_Mean)
print("covarianceof3",classofImage_3_Covariance)
print("meanof7",classofImage_7_Mean)
print("covarianceof7",classofImage_7_Covariance)


# In[12]:


def EvaluationofLikelihood(xvec, meanvec, covmat, dims):
    a = 1 / (((2 * np.pi) ** (dims/2)) * (np.linalg.det(covmat)**(1/2)))
    expon = -(1/2) * np.matmul(np.transpose(xvec-meanvec), np.matmul(np.linalg.inv(covmat), xvec-meanvec))
    expon = np.exp(expon)
    return a * expon


# In[13]:


def PrintProbabilityofError(prior1,prior2, data):
    BayesClassifError=[]
    covar1 = classofImage_3_Covariance
    covar2 = classofImage_7_Covariance

    meanvec1 = classofImage_3_Mean
    meanvec2 = classofImage_7_Mean
    

    for i in range(0,len(data)):
        likelihood1 = EvaluationofLikelihood(data[i], meanvec1, covar1, len(data[i]))
        likelihood2 = EvaluationofLikelihood(data[i], meanvec2, covar2, len(data[i]))
        evidence = likelihood1 * prior1 + likelihood2* prior2
        prob1 = likelihood1 * prior1 / evidence
        prob2 = likelihood2 * prior2 / evidence

        if(prob1<prob2):
            BayesClassifError.append(prob1)
        else:
             BayesClassifError.append(prob2)
    return BayesClassifError


# In[14]:


k=PrintProbabilityofError(0.5,0.5,traindata_new)
np.mean(k)


# In[15]:


k=PrintProbabilityofError(0.5,0.5,testdata_new)
np.mean(k)


# In[16]:


k=PrintProbabilityofError(0.3,0.7,traindata_new)
np.mean(k)


# In[17]:


k=PrintProbabilityofError(0.3,0.7,testdata_new)
np.mean(k)


# In[ ]:




