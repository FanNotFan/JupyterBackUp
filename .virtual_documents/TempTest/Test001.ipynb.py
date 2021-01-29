from scipy.stats import normaltest 
import numpy as np  
import pylab as p 


x1 = np.linspace( -5, 5, 1000 ) 
# print(x1)


mean = x1.mean()
print(mean)


std = x1.std()
print(std)


y1 = np.exp(-((x1 - mean)**2) / (2* std**2)) / (std * np.sqrt(2*np.pi))
# 1./(np.sqrt(2.*np.pi)) * np.exp( -.5*(x1)**2  ) 


p.plot(x1, y1, '.') 


print( '\nNormal test for given data :\n', normaltest(x1)) 


from pandas import DataFrame
from scipy.sparse import csgraph
from scipy.sparse import csr_matrix
x=[a for a in range(10)]
def y_x(x):
    return 2*x**2+4
y=[y_x(i) for i in x]
data=DataFrame({'x':x,'y':y})


data


df_corr = data.corr()


df_corr


np.fill_diagonal(df_corr.values, 0)


df_corr


df_corr>0.99


graph = csr_matrix(df_corr>0.99)


graph


n, labels = csgraph.connected_components(graph)


labels


n



