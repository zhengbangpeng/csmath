# -*- coding: utf-8 -*-
"""
Created on Wed May 07 20:39:42 2014

@author: huajh
"""
from numpy import *
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from numpy.random import uniform, seed

# A sample from book: [book]Machine learning an algorithmic perspective(2009)
    # 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
#Levenberg-Marquardt Algorithm
#
def func_example(x):
    tmp = array([10*(x[1]-x[0]**2),1-x[0]])
    y = dot(transpose(tmp),tmp);
    J = array([[-20*x[0],10],[-1,0]])
    grad = dot(transpose(J),transpose(tmp))
    return y,grad,J
    
def LM(_func,x0,eps = 10**(-7),maxIts = 1000):
    dim = len(x0)
    alpha = 0.01
    y,grad,J = _func(x0)
    it = 0;
    x = x0;
    
    while it < maxIts and linalg.norm(grad) > eps:
        it += 1
        y,grad,J = _func(x)
        H = dot(transpose(J),J) + alpha*eye(dim);
        val,vec = linalg.eig(H)
        while ~(val>zeros(dim)).all():
            alpha *=4;
            H = dot(transpose(J),J) + alpha*eye(dim);
            val,vec = linalg.eig(H);
                    
        it2 = 0;
        while it2 < maxIts:
            it2 +=1
            # solve the least-squares problem - http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
            # lstsq(a,b)
            # x - Least-squares solution. If b is two-dimensional, the solutions are in the K columns of x.
            # residuals - Sums of residuals
            # Rank of matrix a
            # Singular values of a
            Dx,residuals,rank,s = linalg.lstsq(H,-grad)
            xnew = x + Dx;
            ynew,gradnew,Jnew = _func(xnew)
            delta_y = linalg.norm(ynew-y) # actual 
            delta_q = linalg.norm(dot(transpose(grad),xnew-x)) # predicted
            rho = delta_y / delta_q

            if rho > 0:
                x = xnew
                if rho < 0.25:
                    alpha *=4
                elif rho > 0.75:
                    alpha /= 2
                print y,x,linalg.norm(grad),alpha
                break                 
            else:
                alpha *= 4
                print y,x,linalg.norm(grad),alpha
                continue        
    return _func(x0)

def f(x,y): return 100*(y-x**2)**2 + (1-x)**2;

def plot_kindsof():
    y =[1211075609,1211075609 ,606569018.585 ,36089726.2754 ,1998393.70437 ,145602.083316 ,
        78933.3299338 ,19064.0378777 ,0.807285588167 ,0.00377636486519 ,1.12071556816e-10 ,2.36509169373e-16];
    
    x = [[-100,100], [-50.37279597,74.56173802] ,[-25.86272015 , 68.13904581] ,
         [-13.97325118,  53.89512473],[ -7.79661571 , 22.63951628] ,[ -2.49613216 ,-21.86222493] ,
        [  1.2196132 , -12.31978437] ,[ 0.92099293,  0.75872694] ,[ 0.99936359 , 0.9925827 ] ,
        [ 0.99999037 , 0.9999803 ] ,[ 0.99999998, 0.99999997] ,[ 1. , 1.]];
    npts = len(x);
    x0 = array(x)[:,0];
    x1 = array(x)[:,1];
    norm_grad = [16503.7486954,16503.7486954 ,24813470.6353 ,3107968.05969 ,395310.110158 ,
                  59615.5660785 ,14306.743833 ,3640.14001751 ,18.6893703475 ,1.37276917586 ,
                  8.96698008756e-05 ,2.18439819337e-08];
    alpha = [0.01,0.04 ,0.02 ,0.01 ,0.005 ,0.005 ,0.0025 ,0.00125 ,0.000625 ,0.0003125 ,
              0.00015625 ,7.8125e-05];
              
    z = f(x0,x1);

    fig1 = plt.figure(1)   
    plot(x0,x1,'black');
    xi = linspace(-100, 101, 1000)
    yi = linspace(-100, 101, 1000)
    zi = griddata((x0, x1), z, (xi[None,:], yi[:,None]), method='cubic') #
    CS = plt.contour(xi,yi,zi,20,linewidths=0.5,colors='k')
    CS = plt.contourf(xi,yi,zi,20,cmap=plt.cm.jet)
    plt.colorbar() # draw colorbar
    # plot data points.
    plt.scatter(x0,x1,marker='o',c='b',s=10)
    plt.xlim(-105,5)
    plt.ylim(-30,105)
    plt.xlabel('$x_0$');
    plt.ylabel('$x_1$');
    plt.title('$y = (x_1-x_0^2)^2+(1-x_0)^2$ by LMO')
  #  plot(-100,100,'ro',ms=5)
    plot(1,1,'r*',ms=8)
    plt.show()
    fig1.savefig("contour_fig.pdf")

     
if __name__ == '__main__':
    # init 
    #x0 = array([-100,100])    
    #y,grad,J = LM(func_example,x0)        
    
    plot_kindsof();