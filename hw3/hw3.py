# -*- coding: utf-8 -*-
"""
Created on Sun May 04 22:07:59 2014

@author: huajh
"""
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def k_means(data,k):
    num,dim= data.shape
    maxd = [max(data.T[i]) for i in range(dim)];
    mind = [min(data.T[i]) for i in range(dim)];
    means = random.rand(k,dim);
    for i in range(dim):
        means[:,i] = (maxd[i] - mind[i])*means[:,i] + mind[i];
    tuns = 0.01;
    ind = [0 for i in range(num)]
    for it in range(100):
        for i in range(num):        
            closest_k = 0;
            smallest_err = inf;
            for j in range(k):
                dist = linalg.norm(data[i]-means[j]);
                if dist < smallest_err:
                    smallest_err = dist;
                    closest_k = j;
                    ind[i] = j;
                means[closest_k] = means[closest_k]*(1-tuns) + data[i]*(tuns);
    return ind,means
               
def em_gauss_mix(data,k):
    # init    
    idx, means = k_means(data,k);
    (num,dim) = data.shape
    z = zeros([num,k]);
    for i in range(num):
        z[i,idx[i]] = 1;
    alpha = matrix(z);
    data = matrix(data);    
    mu = zeros([k,dim]);
    cov = zeros([k,dim,dim]);
    w = zeros([k,1]);
    loglld = 0;
    eps = 1e-7;
    maxiter = 100;
    it = 0;
    while it < maxiter:
        # M step
        it = it + 1;
        N_k = sum(alpha,0);
        for i in range(k):
            mu[i] = 1/N_k[0,i] * alpha[:,i].T*data;
            cov[i] = 1/N_k[0,i] * multiply(alpha[:,i],(data - mu[i])).T * (data - mu[i]);
            w[i] = N_k[0,i]/num;        
        
        poster = matrix(zeros([num,k]));
        for i in range(k):
            poster[:,i] = w[i,0]*gauss_pdf(data,matrix(mu[i]),matrix(cov[i]))
        #Evaluate the log likelihood
        newloglld = sum(log(sum(poster,1)),0);
        if abs(newloglld-loglld) < eps*abs(newloglld):
            break;
        loglld = newloglld;  
        # E step
        #the numpy equivalent of repmat(a,m,n) is tile(a,(m,n))
        alpha = poster / tile(sum(poster,1),(1,k));  
        
    return alpha,mu,cov,w
    
def gauss_pdf(x, mu,cov):
    num,dim = x.shape;
    if (1,dim) == mu.shape and (dim,dim) == cov.shape:
        det = linalg.det(cov);
        if det == 0:
            raise NameError("The convariance matrix can't be singlar");
        
        norm_cost = 1.0/(math.pow(2*pi, float(dim)/2)*math.pow(det,1.0/2))
        x_mu = matrix(x-mu);        
        return norm_cost * exp(-0.5*sum(multiply(x_mu * cov.I,x_mu),1) )
    else:
        raise NameError("The dimension of the input don't match");        


def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)
    
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
    
if __name__ == '__main__':
        
    N = 200;
    weight = [0.33,0.33,0.34];
    
    mu = [0,0];
    cov = [[0.1,0],[0,1.5]];    
    data1 = random.multivariate_normal(mu,cov,int(N*weight[0]))
    
    mu2 = [4,4];
    cov2 = [[0.1,0],[0,1.5]];
    data2 = random.multivariate_normal(mu2,cov2,int(N*weight[1]))
    
    mu3 = [2,2];
    cov3 = [[1.5,0],[0,0.1]];
    data3 = random.multivariate_normal(mu3,cov3,int(N*weight[2]))
    
    fig1 = plt.figure(1)
    plt.grid(True)
    plt.plot(data1[:,0],data1[:,1],'go');
    plt.plot(data2[:,0],data2[:,1],'ro');
    plt.plot(data3[:,0],data3[:,1],'bo');
    plt.axis('equal');     
    plt.show();
    fig1.savefig("2D_Gauss_Dist.pdf");
    
    data = array(data1.tolist() + data2.tolist() + data3.tolist());    
    (alpha,Mu,Cov,w) = em_gauss_mix(data,3)
    ind = alpha.argmax(1)
    
    fig2 = plt.figure(2)
    plt.grid(True)
    plt.plot(data[find(ind==0),0],data[find(ind==0),1],'go');
    plt.plot(data[find(ind==1),0],data[find(ind==1),1],'bo');
    plt.plot(data[find(ind==2),0],data[find(ind==2),1],'ro');
    plt.axis('equal');
    plot_cov_ellipse(Cov[0], Mu[0], nstd=3, alpha=0.5, color='green')
    plot_cov_ellipse(Cov[1], Mu[1], nstd=3, alpha=0.5, color='blue')
    plot_cov_ellipse(Cov[2], Mu[2], nstd=3, alpha=0.5, color='red')
    plt.show();
    fig2.savefig("EM_MoGs.pdf");
    
    

    

