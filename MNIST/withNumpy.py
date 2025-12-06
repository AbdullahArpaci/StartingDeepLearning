import numpy as np



class MyModel:
    def __init__(self,input_dim,output_dim):
        self.w1 = np.random.randn(128,input_dim)
        self.b1 = np.random.randn(128,1)

        self.w2 = np.random.randn(64,128)
        self.b2 = np.random.randn(64, 1)

        self.w3 = np.random.randn(output_dim, 64)
        self.b3 = np.random.randn(output_dim, 1)
    def ReLU(self,x):
        return np.maximum(0,x)

    def Softmax(self, logits):
        exp_vals = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    def Linear(self,x,w,b):
        return np.dot(x, w) + b
    def forward(self,x):

        z1 = self.Linear(x,self.w1,self.b1)
        a1 = self.ReLU(z1)

        z2 = self.Linear(a1,self.w2,self.b2)
        a2 = self.ReLU(z2)

        results = self.Linear(a2,self.w3,self.b3)

        return results
    def predict(self,x):
        x = self.forward(x)
        probs = self.Softmax(x)

        return probs






