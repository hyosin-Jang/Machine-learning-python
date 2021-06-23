from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data,
													cancer.target,
													test_size=0.3,
													random_state=12)

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# LogisticRegression 및 DecisionTreeClassifier 객체 생성
lr = LogisticRegression(random_state=12)
dt = DecisionTreeClassifier(random_state=12)

# VotingClassifier 객체 생성
voting = VotingClassifier(estimators=[('LR',lr), ('DT',dt)], voting='soft')

# VotingClassifier 학습 및 검증
voting.fit(x_train , y_train)
pred = voting.predict(x_test)

# accuracy_score 호출하여 accuracy 계산 후 출력
print('Accuracy: {0:.3f}'.format(accuracy_score(y_test, pred)))