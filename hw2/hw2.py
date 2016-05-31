__author__ = 'huajh'

from numpy import *
import matplotlib.pyplot as plt

def get_data(filename,num):
    f = open(filename)

    Ylist = []
    count = 0
    for line in f:
        lst = map(int,line.split(','))
       # array1 = np.array(lst[0:-1])        
        gnd = lst[-1]
        if gnd == num:          
            Ylist.append(lst[0:-1])
            count = count + 1
#            if count <2:
#                img = mat(lst[0:-1]).reshape([8,8])
#                img = (255 - img)/255.0
#                plt.imshow(img,cmap=plt.cm.gray)
#                plt.show()
    # d (data dimension) x N (number of data samples)  
    # normalization     
    # return (255-(mat(Ylist)).T)/255.0
    return  ((mat(Ylist)).T)/255.0

def extr_feature(Ymat):
     # p feature dimension
    #     
    p = 2
    Yavg = Ymat.mean(1)
    
    #SVD
   # U, s, V = np.linalg.svd(Ymat-Yavg,full_matrices=False)
   # S = np.diag(s)
    #eigenvectors
   # X = U[0:p,:].T
    #eigvector
   # lmbda = s[0:p]**2
    
    # PCA
    #scatter matrix
    lmbda,Vec = linalg.eig((Ymat-Yavg)*(Ymat-Yavg).T)
    return Yavg, Vec[:,0:p], lmbda[0:p]

def plot_eig(Yavg, Vec,digit,lmbd):
    fig1 = plt.figure(1,figsize=(12, 5))
    plt.subplot(131)
    plt.xlabel('mean')
    plt.imshow(Yavg.reshape(8,8),cmap=plt.cm.gray)
    #plt.axis('off')
    plt.subplot(132)
    plt.xlabel('First Component')
    plt.title('$\lambda = %.3f$' % lmbd[0])
    plt.imshow(Vec[:,0].reshape(8,8),cmap=plt.cm.gray)
    #plt.axis('off')
    plt.subplot(133)
    plt.title('$\lambda = %.3f$' % lmbd[1])
    plt.xlabel('Second Component')
    plt.imshow(Vec[:,1].reshape(8,8),cmap=plt.cm.gray)
    #plt.axis('off')
    plt.subplots_adjust(wspace=0.2, hspace=0.2, top=1, bottom=0, left=0.1, right=0.9)
    plt.show()
    fig1.savefig("mean_compoent_%d.pdf"%digit);

def feature_space(Y, Vec, lmbd):
    dim,N = Y.shape
  #  for d in Ymat[:,0:3].T:
  #      print d
    x = map(lambda dd:lmbd[0]*array(dd*Vec[:,0])[0][0], Y[:,0:N+1].T)
    y = map(lambda dd:lmbd[1]*array(dd*Vec[:,1])[0][0], Y[:,0:N+1].T)
    return x,y

def find_feat_points(x,y,p):
    N = len(x)
    interval_x = (max(x)-min(x))/float(p)
    interval_y = (max(y)-min(y))/float(p)
    ref_x = map(lambda t:min(x)+interval_x*(t+0.5),range(p))
    ref_y = map(lambda t:min(y)+interval_y*(t+0.5),range(p))
    p_indx = []    
    for i in range(p):
        for j in range(p):
            f = lambda a:(ref_x[i]-x[a])**2+(ref_y[j]-y[a])**2            
            dist = map(f,range(N))            
            ind = dist.index(min(dist))
            p_indx.append(ind)            
    return p_indx, ref_x, ref_y
    
def plot_images_matrix(Ymat,indx,digit):
    p = int(sqrt(len(indx)))    
    fig3 = plt.figure(3,figsize=(10, 9))
    for i in range(p):
        for j in range(p):
            plt.subplot(p,p,p*(p-i-1)+j+1)
            img = mat(Ymat[:,indx[i*p+j]].reshape(8,8))
            plt.imshow(img,cmap=plt.cm.gray)
            plt.axis('off')
            #plt.grid(True)
    plt.subplots_adjust(wspace=0.01, hspace=0.02, top=0.95, bottom=0.05, left=0.05, right=0.95)
    plt.show()
    fig3.savefig('images_matrix_%d.pdf'%digit)
    
if __name__ == '__main__':
    
 #   x = range(100)
 #   y = range(100)
 #   plot(x,y,'go')

    digit = 9
    Ymat = get_data('optdigits.tra',digit)
    Yavg, Vec, lmbd = extr_feature(Ymat)
#    img = mat(Ymat[:,0].reshape(8,8))
#    plt.imshow(img,cmap=plt.cm.gray)
#    plt.show()
    
    plot_eig(Yavg,Vec,digit,lmbd)
    
    
    x,y = feature_space(Ymat-Yavg, Vec, lmbd )
    p = 5
    indx,ref_x,ref_y = find_feat_points(x,y,p)
    fig2 = plt.figure(2)
    plt.xlabel('First Compoent')
    plt.ylabel('Second Compoent')
    x = dot(x,100)
    y = dot(y,100)
    plt.plot(x,y,'o',color='#00FF00')
   # plt.grid(True)
    plt.axhline(y=0,color='black')
    plt.axvline(x=0,color='black')
    p = int(sqrt(len(indx)))
    for i in range(p):
        plt.axvline(100*ref_x[i],linestyle='--',color='.5')
        plt.axhline(100*ref_y[i],linestyle='--',color='.5')
        
    plt.plot((array(x))[indx],(array(y))[indx],'ro')
    plt.show()
    fig2.savefig('feature_space_%d.pdf'%digit)    

    plot_images_matrix(Ymat,indx,digit)
    
    