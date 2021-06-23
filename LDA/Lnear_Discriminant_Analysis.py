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

#데이터프레임 생성하기 ==> 학습시키기 위한 데이터(독립변수들)
iris_df = pd.DataFrame(data= iris_data, columns= iris_cols)
print(iris_df.head(5))

#데이터프레임에 학습 데이터의 정답값(라벨 컬럼) 데이터 추가(종속변수)
iris_df['label'] = data['target'] 
print(iris_df.head(5))

#종속변수 각 그룹에 대해 데이터 카운팅 해보기
check_df = iris_df.groupby(by='label').count()

#LDA(선형판별분석) 수행
#LDA 패키지 import
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#학습시키기 위한 독립변수와 종속변수 분할하기
X_train = iris_df[iris_cols]
y_train = iris_df['label']

#LDA 오브젝트생성 및 독립변수와 종속변수를 이용해 LDA 환경구성
lda = LDA().fit(X_train, y_train)

print("판별식 선형계수\n", lda.coef_)
print("판별식 상수\n", lda.intercept_)
y_pred = pd.DataFrame(lda.predict(X_train))
print("예측결과\n", y_pred.head(5))
y_pred_score = pd.DataFrame(lda.predict_proba(X_train))
print("예측스코어\n", y_pred_score.head(5)) #선형판별식에서 얻어지는 예측점수
print("예측정확도\n", lda.score(X_train,y_train))

# 분류 결과 확인하기
from sklearn.metrics import confusion_matrix

conf_df = pd.DataFrame(confusion_matrix(y_train, lda.predict(X_train)))
conf_df.columns=['pred 0', 'pred 1', 'pred 2']#setosa,versicolor,virginica
conf_df.index = ['real 0', 'real 1', 'real 2']
print('Confusion Matrix \n',conf_df)  

#시각화로 확인해보기
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
cld=LinearDiscriminantAnalysis()

X_lda = cld.fit_transform(X_train, y_train)
print(X_lda.shape)

#데이터셋 시각화 해보기
fig, ax = plt.subplots(ncols=2)

sns.scatterplot(iris_df['sep_len'], iris_df['sep_wt'], 
                hue=iris_df['label'], ax=ax[0])
sns.scatterplot(X_lda[:,0], X_lda[:,1], hue=y_train, ax=ax[1])
plt.show()
