

# Implement polynomial curve fitting

from numpy import *
import pylab
#import matplotlib.pyplot as plt

def sin_wgn_sample(num):
    x = linspace(0, 1, num)
    mu = 0
    sigma = 0.15
    #  mu        is the mean of the normal distribution you are choosing from
    # sigma     is the standard deviation of the normal distribution
    # num       is the number of elements you get in array noise
    
    noise = random.normal(mu, sigma, num)
    return x, sin(x*2*pi) + noise

def curve_poly_fit(x, y, M,Reg=0):
    # Least squares solution : w = (P^TP)^{-1} (P^Ty)
    # degree M
    N = x.size
    P = mat(zeros(shape=(M+1,N)))
    
    # The polynomial's coefficients, in decreasing powers
    for i in range(0,M+1):
        col = x**(M-i)
        P[i,:] = col
    w = linalg.inv(P*P.T+Reg*eye(M+1))*(P*(mat(y)).T)
    return w.A1

def main():
    SAMPLE_NUM = 10
    degree = 9
    x, y = sin_wgn_sample(SAMPLE_NUM)
    fig = pylab.figure(1)
    pylab.grid(True)
    pylab.xlabel('x')
    pylab.ylabel('y')
    pylab.axis([-0.1,1.1,-1.5,1.5])

    # sin(x) + noise
    # markeredgewidth mew
    # markeredgecolor mec
    # markerfacecolor mfc

    # markersize      ms
    # linewidth       lw
    # linestyle       ls
    pylab.plot(x, y,'bo',mew=2,mec='b',mfc='none',ms=8)

    # sin(x)
    x2 = linspace(0, 1, 1000)
    pylab.plot(x2,sin(2*x2*pi),'#00FF00',lw=2,label='$y = \sin(x)$')

    # polynomial fit
    reg = exp(-18)
    w = curve_poly_fit(x, y, degree,reg) #w = polyfit(x, y, 3)
    po = poly1d(w)      
    xx = linspace(0, 1, 1000)
    pylab.plot(xx, po(xx),'-r',label='$M = 9, \ln\lambda = -18$',lw=2)
    
    pylab.legend()
    pylab.show()
    fig.savefig("poly_fit9_10_reg.pdf")

if __name__ == '__main__':
    main()
