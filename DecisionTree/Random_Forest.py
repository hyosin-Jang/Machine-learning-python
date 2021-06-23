from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data,
													cancer.target,
													test_size=0.3,
													random_state=12)

# RandomForestClassifier 임포트
from sklearn.ensemble import RandomForestClassifier

# RandomForestClassifier 객체 생성
rf = RandomForestClassifier(random_state=12)

# RandomForestClassifier 객체 학습 및 검증
rf.fit(x_train, y_train)
pred = rf.predict(x_test)

print('Accuracy: {0:.3f}'.format(accuracy_score(y_test, pred)))