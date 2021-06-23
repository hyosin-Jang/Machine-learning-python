# K Fold Validation을 위한 cross_val_score() 메서드 로딩
from sklearn.model_selection import cross_val_score

# 모델 구현을 위한 라이브러리 로딩
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
import numpy as np

# load_iris() 메서드를 이용하여 iris 데이터 셋 로드
iris = load_iris()
data = iris.data
label = iris.target

# DecisionTreeClassifier 모델 객체 생성
dt_clf = DecisionTreeClassifier(random_state=156)

# cross_val_score() 메서드를 이용하여 교차 검증 수행
scores = cross_val_score(estimator=dt_clf,X=data, y=label, scoring='accuracy', cv=3 )

# 교차 검증 수행 결과 성능 지표 출력
print('Fold val accuracy:',np.round(scores, 4))
print('Avg val accuracy:', np.round(np.mean(scores), 4))