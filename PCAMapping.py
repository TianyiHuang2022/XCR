import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as pyplot
from pandas import DataFrame


hidden = np.load('embedded_fashionz.npy')
center = np.load('center_fashion.npy')
Y = np.load('fashion_y.npy')
n_clusters = 10




pdata = np.vstack((hidden, center))
pca = PCA(n_components=2)
pca_f = pca.fit_transform(pdata)

shape = pca_f.shape
shape = shape[0]

a = pca_f[0:shape-n_clusters,0]
b = pca_f[0:shape-n_clusters,1]
df = DataFrame(dict(x=a, y=b, label=Y))
colors = {0: 'green', 1: 'yellow', 2: 'deeppink',  3: 'purple', 4: 'cyan', 5: 'olive', 6: 'darkblue', 7: 'orange',  8: 'red',
          9: 'darkcyan', 10: 'brown', 11: 'grey', 12: 'black'}
fig, axx = pyplot.subplots()
grouped = df.groupby('label')
indexl = 0
for key, group in grouped:
    group.plot(ax=axx, kind='scatter', s=0.001, x='x', y='y', marker='h', color=colors[key])
pyplot.scatter(pca_f[shape-n_clusters:shape, 0], pca_f[shape-n_clusters:shape, 1], c='black', s=20, marker='*')
pyplot.axis('equal')
pyplot.axis('off')
fig.set_size_inches(5, 5)
pyplot.savefig('PCA_Fashion.png',bbox_inches='tight', dpi=300, pad_inches=0)
pyplot.show()
