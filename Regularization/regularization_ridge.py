# 필요한 라이브러리 로딩
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
# 릿지 회귀 모델 적용을 위해 Ridge 로딩
from sklearn.linear_model import Ridge

# 데이터셋 로딩
boston = load_boston()

# 데이터셋 분할
x_train, x_test, y_train, y_test = train_test_split(boston.data,boston.target,test_size=0.3,random_state = 12)

# 규제를 위한 alpha 값 초기화
# 알파값 변경해가면서 확인해보기
alpha = 0.1

# Rigde 클래스 객체 생성
ridge = Ridge(alpha = alpha)

# 규제 학습 수행
ridge.fit(x_train, y_train)

# 모델을 통한 예측
ridge_pred = ridge.predict(x_test)

# 학습된 모델에 대한 R^2 계산
# predict() 학습된 모델 기반으로 테스트 데이터셋 예측 수행 및 결과 반환
# score(): 학습된 모델 기반으로 테스트 데이터셋에 대한 R2 값 계산
r2_train = ridge.score(x_train, y_train)
r2_test = ridge.score(x_test,y_test)
print('Training-datasset R2 : {0:.3f}'.format(r2_train))
print('Test-datasset R2 : {0:.3f}'.format(r2_test))

# 컬럼별 회귀계수 저장한 Series 객체 생성 및 출력              
ridge_coef_table = pd.Series(data=np.round(ridge.coef_,1), index=boston.feature_names)
print('Ridge Regression Coefficients :')
print(ridge_coef_table.sort_values(ascending=False))

# 막대그래프 시각화 
plt.figure(figsize=(10,5))
ridge_coef_table.plot(kind='bar')
plt.ylim(-12,5)
plt.show()