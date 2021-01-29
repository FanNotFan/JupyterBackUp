import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris=load_iris()
X=iris.data
y=iris.target
X.shape,y.shape


import pandas as pd
pd_X = pd.DataFrame(X)
pd_X.head(10)


y


#调用pca
pca = PCA(n_components=2) #实例化
pca = pca.fit(X) #拟合模型
X_dr = pca.transform(X) #获取新矩阵
#X_dr=PCA(n_components=2).fit_transform(X)
X_dr.shape#降维2维了 #(150, 2)


X_dr


X_dr[y==0,0] #布尔索引，y==0代表第一个标签，0代表第一个特征（0列）


plt.figure()
plt.scatter(X_dr[y==0, 0], X_dr[y==0, 1],c='red',label=iris.target_names[0])
plt.scatter(X_dr[y==1, 0], X_dr[y==1, 1], c="black", label=iris.target_names[1])
plt.scatter(X_dr[y==2, 0], X_dr[y==2, 1], c="orange", label=iris.target_names[2])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()


pca_mle = PCA(n_components="mle")
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)
print(X_mle.shape)
print(X_mle.shape[1])
#可以发现，mle为我们自动选择了3个特征

pca_mle.explained_variance_ratio_.sum()
#得到了比设定2个特征时更高的信息含量，对于鸢尾花这个很小的数据集来说，3个特征对应这么高的信息含量，并不需要去纠结于只保留2个特征，毕竟三个特征也可以可视化


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# get_ipython().run_line_magic("matplotlib", " notebook")
get_ipython().run_line_magic("matplotlib", " inline")
# get_ipython().run_line_magic("matplotlib", " widget")

x1, y1, z1 = X_mle[y==0, 0], X_mle[y==0, 1], X_mle[y==0, 2]
x2, y2, z2 = X_mle[y==1, 0], X_mle[y==1, 1], X_mle[y==1, 2]
x3, y3, z3 = X_mle[y==2, 0], X_mle[y==2, 1], X_mle[y==2, 2]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x1, y1, z1, c='y')  # 绘制数据点
ax.scatter(x2, y2, z2, c='r')
ax.scatter(x3, y3, z3, c='g')

ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()


pca_mle = PCA(n_components=0.97, svd_solver="full")
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)
print(X_mle.shape)
print(X_mle.shape[1])
#可以发现，mle为我们自动选择了3个特征

pca_mle.explained_variance_ratio_.sum()
