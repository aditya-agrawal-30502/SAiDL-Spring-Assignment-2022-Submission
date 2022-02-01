import numpy as np
from sklearn.metrics import accuracy_score

def transition(a):
    b = []
    for i in a:
        b.append(np.random.randn(i.shape[0],i.shape[1])+i)
    return b

def likelihood(y,y_pred):
    return -np.sum((y-y_pred)**2)     

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return (x+np.abs(x))/2

def prior(w):
    sume = 0
    for i in w:
        sume+= -np.sum(i**2)
    return sume
class Network():
    def __init__(self):
        self.params = [np.random.randn(2,3),np.random.randn(1,3),np.random.randn(3,3),np.random.randn(1,3),np.random.randn(3,1),np.random.randn(1,1)]
    def __call__(self,x,w = None):
        if w is None:
            w = self.params
        x = x@w[0]+w[1]
        x = sigmoid(x)
        x = x@w[2]+w[3]
        x = sigmoid(x)
        x = x@w[4]+w[5]
        x = sigmoid(x)
        return x

network = Network()
def hasting(data,epochs):
    x,y = data
    accepted = 0
    for i in range(epochs):
        w = network.params
        w_new = transition(w)
        y_pred = network(x,w)
        y_new_pred = network(x,w_new)
        y_pred = y_pred.squeeze()
        y_new_pred = y_new_pred.squeeze()
        w_likelihood_new = likelihood(y,y_new_pred)+prior(w_new)
        w_likelhood = likelihood(y,y_pred)+prior(w)

        if w_likelihood_new>w_likelhood:
            final_weights = w_new
            accepted+=1
        else:
            seed = np.random.uniform()
            if np.log(seed)<w_likelihood_new-w_likelhood:
                final_weights = w_new
                accepted+=1
            else:
                final_weights = w
        network.params = final_weights
        print("epoch:",i)
        print(accuracy_score(train[1],network(train[0])>0.5))
    print(accepted)

w = np.random.uniform(0.0,1.0,[5000,2])
wt = np.round(w)
wt = wt.astype(int)
wt = wt[:,0]^wt[:,1]
train = [w[:4500],wt[:4500]]
test = [w[4500:],wt[4500:]]
hasting(train,10000)
print(accuracy_score(test[1],network(test[0])>0.5))
