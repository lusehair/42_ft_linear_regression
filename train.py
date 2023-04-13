#8499 
# -0.0214 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=1e-4, max_iter=1e5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas
    
    
    
    def add_intercept(self, x):
        """Adds a column of 1’s to the non-empty numpy.array x.
        Args:
        x: has to be a numpy.array of dimension m * n.
        Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """ 
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        b = np.ones(len(x))
        return np.c_[b,x]
    
    def simple_gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.
        Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
        Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """

        # if not isinstance(x, np.ndarray) or x.size == 0 : 
        #     return None 
        # if not isinstance(y, np.ndarray) or y.size == 0 : 
        #     return None 
        # if not isinstance(theta, np.ndarray) or theta.size == 0 : 
        #     return None 
        # if x.shape != y.shape or theta.shape != (2, 1) : 
        #     return None 
        # if x.shape != (x.shape[0], 1) or y.shape != (y.shape[0], 1) : 
        #     return None 

        m = np.size(x)
        # gradient0 = (1/m) * np.sum(theta[0] + (theta[1] * x) - y)
        # gradient1 = (1/m) * np.sum((theta[0]+ (theta[1] * x) - y) * x)
        # theta[0] = gradient0
        # theta[1] = gradient1 
        tmp_theta = np.array([0.0, 0.0]).reshape(2, 1)
        theta = self.thetas 

        for j in range(m):
            tmp_theta[0] += (theta[0] + theta[1] * x[j]) - y[j]
            tmp_theta[1] += ((theta[0] + theta[1] * x[j]) - y[j]) * x[j]
        self.thetas = theta
        return tmp_theta 
            
   



    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exceptions.
        """
        
        if not isinstance(x, np.ndarray) or x.size == 0 : 
            return None 
        
        theta = self.thetas 
        if not isinstance(theta, np.ndarray) or theta.size == 0 : 
            return None 
        if x.shape != (x.shape[0], 1) or theta.shape != (2, 1) : 
            return None 
        
       
        theta0 = theta[0] 
        theta1 = theta[1] 
        m = np.size(X)
        predicted = np.zeros(m)
        for i in range(m):
            predicted[i] = theta0 + (theta1 * X[i])
        return predicted


    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exception.
        """

       
        max_iter = int(self.max_iter)
        alpha = self.alpha 
        # if len(x.shape) != 2 :
        #     x = x.reshape(-1, 1) 
        # if len(y.shape) != 2 :
        #     y = y.reshape(-1, 1) 
        # if len(theta.shape) != 2 :
        #     theta = theta.reshape(-1, 1)
       
        for i in range(max_iter) :
          #  theta = self.thetas
            n = self.simple_gradient(x, y)
            self.thetas -= (alpha * n) / len(y)
        return self.thetas

    def loss_elem_(self,y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """ 
        if not isinstance(y, np.ndarray) or y.size == 0 : 
            return None 
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0 : 
            return None 
        if y.shape != y_hat.shape : 
            return None
        
        ret = [] 

        for (y_el, hat_el) in zip(y, y_hat) : 
            ret = np.append(ret, (hat_el - y_el) ** 2) 
        ret = np.array(ret)
        return ret 


    def loss_(self, y, y_hat):
        """
        Description:
        Calculates the value of loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """ 
        if not isinstance(y, np.ndarray) or y.size == 0 : 
            return None 
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0 : 
            return None 
        if y.shape != y_hat.shape :
            return None
        loss_array = self.loss_elem_(y, y_hat) 
        total = np.sum(loss_array) 
        loss = total / (2 * y.shape[0])
        return loss 

def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x’ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """ 

    if not isinstance(x, np.ndarray) or x.size == 0 : 
        return None 
    if x.shape != (x.shape[0], 1) :
        return None 

    minus = min(x) 
    maxus = max(x) 

    ret = np.zeros(x.shape)
    for i, el in np.ndenumerate(x) : 
        ret[i] = (el -minus) / (maxus - minus) 
    return  ret


def plot(x, y, theta) :

    plt.scatter(x, y, color = "blue") 
    plt.plot(x, theta[0] + theta[1] * x, color = "red") 
    plt.savefig("plot.png")
    plt.show() 

if __name__ == "__main__" : 

    # data = pd.read_csv("data.csv") 
    data = np.loadtxt("data.csv", dtype = np.longdouble, delimiter = ',', skiprows = 1)

    # X = np.array(data.loc[:,"km"]).reshape(-1,1)
    # Y = np.array(data.loc[:,"price"]).reshape(-1,1)  

    # X0 = np.array(data.loc[:,"km"]) 
    # Y0 = np.array(data.loc[:,"price"]) 
    # X0 = X
    # Y0 = Y

   # X = minmax(X) 
    # Y = minmax(Y) 
    X0 = data[:, 0]
    Y0 = data[:, 1]
    Y = Y0
    X = (data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0])
    # Y = (data[:, 1] - np.mean(data[:, 1])) / np.std(data[:, 1])

    myLr = MyLinearRegression(np.array([[0.], [0.]]), 0.3, 1500) 
    # myLr.predict_(X)
    myLr.fit_(X, Y) 

    myLr.thetas[0] -= myLr.thetas[1] * np.mean(X0) / np.std(X0)
    myLr.thetas[1] /= np.std(X0)

    print(myLr.thetas[0].round(2))
    print(myLr.thetas[1].round(2))
    # myLr.thetas = (myLr.thetas * (max(X) - min(X))) + min(X)  
    # myLr.thetas[1] = (myLr.thetas[1] * (max(Y0) - min(Y0))) + min(Y0) 


    #y_hat = myLr.predict_(X) 
    #print(y_hat)
    # print(myLr.thetas)
    plot(X0, Y0, myLr.thetas) 



