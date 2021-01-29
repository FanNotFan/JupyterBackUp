from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=60)
X = faces.data


pca = PCA(150)#实例化
X_dr = pca.fit_transform(X)
print(X_dr.shape)
X_inverse = pca.inverse_transform(X_dr)
print(X_inverse.shape)


#可视化
fig,ax = plt.subplots(2,10,figsize=(10,2.5),subplot_kw={"xticks":[],"yticks":[]})

for i in range(10):
    ax[0,i].imshow(faces.images[i,:,:],cmap="binary_r")
    ax[1,i].imshow(X_inverse[i].reshape(62,47),cmap="binary_r")



