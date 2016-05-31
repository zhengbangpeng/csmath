# -*- coding: utf-8 -*-
"""
Created on Sat May 10 11:09:26 2014

@author: huajh
"""

from numpy import *

class svmclass:
    def __init__(self,data,label):
        self.trainX = data
        self.trainY = label
        self.alpha  = ones([label.shape[0],1])
        self.bias = 0
        self.shift = zeros([1,data.shape[1]])
        self.scalefactor = zeros([1,data.shape[1]])
        self.sv = data
        self.sv_idx = array(range(data.shape[0]))
        self.kernelfunc = rbf_kernel
            

def CreateSampleData():
    
    rad = sqrt(random.rand(100,1)); # Radius
    Ang = 2*math.pi*random.rand(100,1); #Angle
    data = [];
    label = [];
    return data,label;

def linear_kernel(x,y):
    # x,y num  x dim
    return dot(x,y.T)
 
def poly_kernel(x,y):
    gamma = 0.5;
    r = 1;
    d = 3;
    return (gamma*dot(x,y.T)+r)**d
def rbf_kernel(x,y):
    sigma = 0.5;
    return exp( -(1/(2*sigma**2))* ( array([sum(x**2,1)]).T +  array([sum(y.T**2,0)]) - 2*dot(x,y.T) ))

def exp_kernel(x,y):
    sigma = 0.5;
    kval = zeros([x.shape[0],y.shape[0]]);
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            kval[i,j] = sum(abs(x[i,:] - y[j,:]))
    return exp( -(1/(2*sigma**2)) * kval )


def trainsvm(data,label):
    
    label[label<0] = -1
    label[label>=0] = 1    
    ker_func = exp_kernel;
    num,dim = data.shape;
    BoxC = ones([num,1]);
    
    #
    methods = 'LS';
     
    # scale the training data
    shift = mean(data,0);
    scalefactor = 1/std(data,0);
    for i in range(dim):
        data[:,i] = scalefactor[i]*(data[:,i]-shift[i])

    if methods == 'QP':        
        Q = dot(dot(label,label.T),ker_func(data,data))
        e = -ones([num,1]);
        A = label.T
        b = zeros([num,1]);        
        QP(Q,e,A,b,C);
        
    elif methods == 'LS':        
        # %   min C \sum_i \xi_i (or \xi_i^2) + 1/2 |w|^2 
        # kernel
        Omega = ker_func(data,data)
        Omega = 0.5*(Omega+Omega.T) + diag(1/BoxC[:,0])
        
        # hessian matrix ZZ'
        Q = dot(label,label.T)*Omega
        
        # solve Ax = b
        A = matrix(zeros([num+1,num+1]))
        A[1:,0] = matrix(label)
        A[0,1:] = matrix(-label.T)
        A[1:,1:] = matrix(Q);
        b = matrix(ones([num+1,1]))        
        b[0] = 0
        x = linalg.solve(A,b);
        bias = x[0];
        alpha = label*array(x[1:])
        sv = data
        sv_idx = array(range(num))
    
    svm = svmclass(data,label)
    svm.kernelfunc = ker_func
    svm.scalefactor = scalefactor
    svm.shift = shift
    svm.alpha = alpha
    svm.bias = bias    
    svm.sv = sv
    svm.sv_idx = sv_idx
    
    return svm
def QP(Q,e,A,b,C):
    # sovle 
    # min f(x) = 1/2 x^T Q x + e^T x
    # st. Ax = b
    H  = 1

def classifysvm(svm,testX):    
    # shift and rescale data
    for i in range(testX.shape[1]):
        testX[:,i] = svm.scalefactor[i]*(testX[:,i]-svm.shift[i])
    
    pred = dot((svm.kernelfunc(svm.sv, testX)).T,svm.alpha) + svm.bias
    
    # recovery
    for i in range(testX.shape[1]):
       testX[:,i] = (1/svm.scalefactor[i])*testX[:,i]+svm.shift[i]
       
    return sign(pred)    
    
if __name__ == '__main__':
   # data,label = CreateSampleData();
    num = 100;
    rad1 = sqrt(random.rand(num,1)); # Radius
    ang1 = 2*math.pi*random.rand(num,1); #Angle
    data1 = zeros([num,2]);
    data1[:,0] = (rad1*cos(ang1))[:,0]
    data1[:,1] = (rad1*sin(ang1))[:,0]
    
    rad2 = sqrt(3*random.rand(num,1)+1); # Radius
    ang2 = 2*math.pi*random.rand(num,1); #Angle
    data2 = zeros([num,2]);
    data2[:,0] = (rad2*cos(ang2))[:,0]
    data2[:,1] = (rad2*sin(ang2))[:,0]
    
    fig1 = plt.figure(1)
    plt.grid(True)
    h1 = plt.plot(data1[:,0],data1[:,1],'bo',label="1(train)")
    h2 = plt.plot(data2[:,0],data2[:,1],'ro',label="2(train)");
    
    cir = plt.Circle((0, 0), 1, facecolor='none',edgecolor='g', linewidth=2, alpha=0.5)
    plt.gca().add_patch(cir)
    cir = plt.Circle((0, 0), 2, facecolor='none',edgecolor='r', linewidth=2, alpha=0.5)
    plt.gca().add_patch(cir)    
    plt.axis('equal')           
    
    traindata = array(data1.tolist() + data2.tolist()); 
    label = label = ones([2*num,1])
    label[num:2*num] = -1
    
    testnum = 100
    rad0 = sqrt(4*random.rand(testnum,1)); # Radius
    ang0 = 2*math.pi*random.rand(testnum,1); #Angle
    testdata = zeros([testnum,2]);
    testdata[:,0] = (rad0*cos(ang0))[:,0]
    testdata[:,1] = (rad0*sin(ang0))[:,0]
    plt.plot(testdata[:,0],testdata[:,1],'bo',mfc='none');
        
    # two class
    
    svm = trainsvm(traindata,label);
    labelY = classifysvm(svm,testdata)    
    idx1 = find(labelY == 1)
    idx2 = find(labelY == -1)
    plt.plot(testdata[idx1,0],testdata[idx1,1],'bx',label="1(classified)");
    plt.plot(testdata[idx2,0],testdata[idx2,1],'rx',label="2(classified)");
    plt.legend()


    truelabel = ones([testnum,1])
    for i in range(testnum):
        dist = sqrt(sum(testdata[i,:]**2,0))
        if dist >1:
            truelabel[i] = -1        

    accuracy = (testnum-0.5*sum(abs(truelabel-labelY),0))/testnum
    
    plt.title('exp - Accuracy = %.2f'%accuracy[0,0]);
    plt.show()
    fig1.savefig("hw5_exp_rbf_classified.pdf");
