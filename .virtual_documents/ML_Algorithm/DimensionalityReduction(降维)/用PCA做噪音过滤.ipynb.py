from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import numpy as np

digits = load_digits()
print(digits.data.shape)

def plot_digits(data):
    fig,axes = plt.subplots(4,10,figsize=(10,4)
                           ,subplot_kw = {"xticks":[],"yticks":[]})
    for i,ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),cmap="binary")

#可视化
plot_digits(digits.data)


#加入噪声
rng = np.random.RandomState(42)
noisy = rng.normal(digits.data,2)
plot_digits(noisy)


#过滤噪声
pca = PCA(0.5,svd_solver="full").fit(noisy)
X_dr = pca.transform(noisy)
print(X_dr.shape)

without_noise = pca.inverse_transform(X_dr)
plot_digits(without_noise)



