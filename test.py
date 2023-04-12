import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    thetas = np.loadtxt('thetas.csv', delimiter=',')
    data = pd.read_csv("data.csv") 
    X = np.array(data.loc[:,"km"]).reshape(-1,1)
    Y = np.array(data.loc[:,"price"]).reshape(-1,1)  


    plt.axes().grid() 
    xt = np.linspace(1000, max(X), 100) 
    yt =  thetas[1] * xt + thetas[0]
    plt.plot(xt, yt, 'b-', label='Hypothesis: h(x) = %0.2f + %0.2f x' % (thetas[0], thetas[1])) 
    plt.plot(X, Y, 'ro', label='Training data') 
    plt.legend()
   # plt.plot(-0.021, 8499) 
    # plt.plot(thetas[0], thetas[1]) 
    plt.savefig('test.png')



    