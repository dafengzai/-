import numpy as np
import pandas as pd
import visuals as vs
from IPython.display import display # 使得我们可以对DataFrame使用display()函数

data = pd.read_csv("customers.csv")
data.drop(['Region', 'Channel'], axis = 1, inplace = True)
print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)

# TODO：从数据集中选择三个你希望抽样的数据点的索引
indices = [40, 100, 250]
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)



# TODO：使用自然对数缩放数据
log_data = np.log(data)

# TODO：使用自然对数缩放样本数据
log_samples = np.log(samples)

# 可选：选择你希望移除的数据点的索引
outliers  = [154, 65, 75, 66, 128]

from collections import Counter
cnt = Counter()
for feature in log_data.keys():
     Q1 = np.percentile(log_data[feature], 25)
    
    # TODO：计算给定特征的Q3（数据的75th分位点）
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO：使用四分位范围计算异常阶（1.5倍的四分位距）
    step = 1.5*(Q3-Q1)
    cnt[feature] += log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index


# 如果选择了的话，移除异常点
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

# TODO：通过在good data上使用PCA，将其转换成和当前特征数一样多的维度
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)

# TODO：通过在good data上使用PCA，将其转换成和当前特征数一样多的维度
from sklearn.decomposition import PCA
pca = PCA(n_components = 6)

# TODO：使用上面的PCA拟合将变换施加在log_samples上
pca_samples = pca.fit(log_samples)

pca_results = vs.pca_results(good_data, pca)

# TODO：通过在good data上进行PCA，将其转换成两个维度
pca = PCA(n_components = 2)

# TODO：使用上面训练的PCA将good data进行转换
reduced_data = pca.fit_transform(good_data)

# TODO：使用上面训练的PCA将log_samples进行转换
pca_samples = pca.transform(log_samples)

# 为降维后的数据创建一个DataFrame
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
vs.biplot(good_data, reduced_data, pca)

# TODO：在降维后的数据上使用你选择的聚类算法
from sklearn.mixture import GaussianMixture
clusterer = GaussianMixture(n_components = 2, random_state = 66)
clusterer.fit(reduced_data)

# TODO：预测每一个点的簇
preds = clusterer.predict(reduced_data)
print(preds)