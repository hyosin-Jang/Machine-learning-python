from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data,
													cancer.target,
													test_size=0.3,
													random_state=12)

# Hyperparameter 설정 없이 학습
dt = DecisionTreeClassifier(random_state=12)
dt.fit(x_train, y_train) # x_train(feature, 독립변수)과 y_train(calss, 종속변수)로 학습(train)시켜라!

# max depth와 leaf 노드 개수 확인, get_depth(), get_n_leaves() 함수 사용
print("Max Depth: ", dt.get_depth())
print("Number of leaves: ", dt.get_n_leaves())

# max_depth를 3으로 설정 후 학습
dt = DecisionTreeClassifier(max_depth=3, random_state=12)
dt.fit(x_train, y_train)

# max depth와 leaf 노드 개수 확인, get_depth(), get_n_leaves() 함수 사용
print("Max Depth: ", dt.get_depth())
print("Number of leaves: ", dt.get_n_leaves())

# max_leaf_nodes를 9으로 설정 후 학습
dt = DecisionTreeClassifier(max_leaf_nodes=9, random_state=12)
dt.fit(x_train, y_train)

# max depth와 leaf 노드 개수 확인, get_depth(), get_n_leaves() 함수 사용
print("Max Depth: ", dt.get_depth())
print("Number of leaves: ", dt.get_n_leaves())