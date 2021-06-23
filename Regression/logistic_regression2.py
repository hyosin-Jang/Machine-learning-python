# 필요한 데이터셋 로딩
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

# 데이터셋 로딩
cancer = load_breast_cancer()
x=cancer.data
y=cancer.target
# StandardScaler() 활용한 데이터 스케일링 
scaler = StandardScaler()
scaler.fit(x)
data_scaled = scaler.transform(x)

# 학습데이터와 테스트 데이터로 분할​
x_train, x_test, y_train, y_test = train_test_split(data_scaled, 
                                                    cancer.target, 
                                                    test_size=0.3, 
                                                    random_state=12)

# 로지스틱 회귀 분석 모델 생성 및 학습
clf = LogisticRegression()
clf.fit(x_train,y_train)

# 학습된 모델에 테스트 데이터(x_test) 입력하여 예측값 생성
y_pred = clf.predict(x_test)

# 오차 행렬 생성
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix')
print(confusion)

# 정확도, 정밀도, 재현율 계산 
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
  
print('Accuracy: {0:.4f},Precision: {1:.4f},Recall: {2:.4f}'
      .format(accuracy , precision ,recall))