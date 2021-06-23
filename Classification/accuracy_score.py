from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data,
													cancer.target,
													test_size=0.3,
													random_state=12)

dt = DecisionTreeClassifier(random_state=12)
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)

# 성능 지표 측정 함수 임포트 - accuracy_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_score,recall_score

# accuracy 계산
accuracy = accuracy_score(y_test, y_pred)
# precision 계산
precision = precision_score(y_test, y_pred)
# recall 계산
recall = recall_score(y_test, y_pred)

# 성능 지표 출력
print("Accuracy: {:.3f}".format(accuracy))
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))