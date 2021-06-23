# 필요한 라이브러리 로딩
import pandas as pd
import numpy as np

# StandardScaler, train_test_split, LogisticRegression 로딩
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 분류 모델을 위한 성능 지표 함수 로딩
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# URL 통해서 캐글의 자전거 대여 수요 데이터셋 다운로드
url = 'https://codepresso-online-platform-public.s3.ap-northeast-2.amazonaws.com/learning-resourse/python-machine-learning-20210326/bike-demand.csv'
df_bike = pd.read_csv(url)

# 독립변수 데이터 생성
# temp, atemp, humidity, windspeed	컬럼 데이터만 저장
X_df_bike = df_bike.iloc[:, 5:9]
# print(X_df_bike.head(5))

# 종속변수 데이터를 위한 파생변수 생성
# 총 대여건수(count) 가 500 이상인 경우 1, 미만인 경우 0
df_bike['y'] = 1
df_bike.loc[df_bike['count'] < 500, 'y'] = 0
y = df_bike['y'] 

# StandardScaler 이용한 스케일링
scaler = StandardScaler()
scaler.fit(X_df_bike)
result = scaler.transform(X_df_bike)

# 스케일된 결과 데이터를 DataFrame 으로 저장
X_scaled_bike = pd.DataFrame(data=result, columns=X_df_bike.columns)

# 데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(X_scaled_bike, y, test_size=0.3, random_state = 12)

# LogisticRegression 모델 객체 생성
clf = LogisticRegression()

# 훈련 데이터를 통한 학습
clf.fit(x_train,y_train)

# 학습된 모델에 테스트 데이터셋 이용하여 예측값 생성
y_pred = clf.predict(x_test)

# score 메소드를 통한 정확도 측정
train_score = clf.score(x_train, y_train)
test_score = clf.score(x_test, y_test)
print('Training Data Accuracy: {:0.3f}'.format(train_score))
print('Testing Data Accuracy: {:0.3f}'.format(test_score))

# 오차 행렬 생성
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrixs : \n', confusion)

# 정확도, 정밀도, 재현율 계산 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
  
print('Accuracy: {0:.4f}, Precision: {1:.4f}, Recall: {2:.4f}'
      .format(accuracy , precision ,recall))