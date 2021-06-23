from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#DecisionTreClassifier 임포트
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data,cancer.target, test_size=0.3, random_state=12)


#DecisionTreClassifier 객체 생성
dt = DecisionTreeClassifier(random_state=12)
#fit 함수로 Decision Tree 모델 학습
dt.fit(x_train, y_train)
#학습된 Tree의 Depth 확인 - get_Depth() 함수 사용
print("Depth of tree :", dt.get_depth())

#학습된 Tree의 리프 노드 개수 확인 - get_n_leaves() 함수 사용
print("Number of leaves: ", dt.get_n_leaves())

#테스트 세트 라벨 예측
y_pred = dt.prediction(x_test)

#accuracy, precision, recall 계산
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_pred, y_test)
recall = recall_score(y_pred, y_test)

#성능 지표 출력
print("Accuracy: {: .3f}" ,format(accuracy))
print("Precision: {: .3f}" ,format(precision))
print("Recall: {: .3f}" ,format(recall))