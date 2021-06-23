from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#iris 데이터셋 로드와 Dict 포맷의 키 확인하기
data = load_iris()
print("iris dataset format and keys\n",data.keys())

#feature name과 관측값 가져오기
iris_data = data['data']
iris_cols = data['feature_names']
print("iris dataset columns\n",iris_cols)

#column name을 재설정
iris_cols=['sep_len', 'sep_wt', 'pet_len', 'pet_wt']

#데이터프레임 생성
iris_df = pd.DataFrame(data= iris_data, columns= iris_cols)
iris_df['target'] = data['target']
print(iris_df.head(5))

#품종별 데이터 카운팅 체크
target_cnt_df = iris_df.groupby(by='target').count()
print(target_cnt_df)

#PCA 수행
#표준화(스케일링)
from sklearn.preprocessing import StandardScaler

X_train = iris_df.iloc[:, :4]
iris_z_score = StandardScaler().fit_transform(X_train) 

iris_z_df = pd.DataFrame(data= iris_z_score, columns= iris_cols)
print(iris_z_df.head(5))

#주성분 분석(Feature 4 --> 2 axes)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

pca.fit(iris_z_df)

#주성분 찾기 : 고유벡터
print('PCA Shape:\n', pca.components_.shape)
print('PCA eigenvectors:\n',pca.components_)

#고유벡터에 데이터를 투영시키는 과정이 transform이다.
X_pca = pca.transform(iris_z_df)
print('PCA Projection result(shape)\n',X_pca.shape)

#각 주성분이 분산을 얼마나 잘 설명하는지를 나타냄
import numpy as np
print('variance :\n',pca.explained_variance_ratio_)
print('total variance :\n', np.sum(pca_explained_variance_ratio_))
print('\n')

#projection 된 결과를 데이터프레임으로 구성
pca_cols = ['pca_com_1', 'pca_com_2']
pca_df = pd.DataFrame(data= X_pca, columns= pca_cols)
pca_df['target'] = data['target']
print(pca_df.head(5))

#주성분 분석 결과 시각화
fig, ax = plt.subplots(ncols=2)

sns.scatterplot(iris_df['sep_len'], iris_df['sep_wt'], 
                hue=iris_df['target'], ax=ax[0])
sns.scatterplot(pca_df['pca_com_1'], pca_df['pca_com_2'], 
                hue=pca_df['target'], ax=ax[1])
plt.show()
