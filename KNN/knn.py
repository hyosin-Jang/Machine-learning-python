from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 1. 사이킷런의 지도학습 모델 라이브러리
from sklearn.neighbors import KNeighborsClassifier

# import 잘 됐는지 확인
#print(sklearn.__version__)

# 2. 데이터셋 로딩
iris = load_iris()

# DESCR 속성 이용하여 데이터셋의 전체적인 정보 확인
#print(iris.DESCR)

'''
feature_names로 feature에 접근 print(iris.feature_names)
data로 data에 접근 print(iris.data)
target_names로 target의 종류 확인 print(iris.target_names)
'''
# target로 target 확인
#print(iris.target)


# 속성 이용하여 feature 확인, shape는 행, 열 개수 리턴
print('Iris data shape:', iris.data.shape)
print('Iris feature name\n:', iris.feature_names)
print('Iris data\n:', iris.data)
print('Iris data type\n:', type(iris.data))

# 속성 이용하여 class 확인
print('iris target name:\n',iris.target_names)
print('iris target value:\n',iris.target)

# 3. train_test_split함수로 데이터셋을 train, test 로 분할, feature는 x, class는 y
# random_state를 같은 값으로 설정하면, 실행 때마다 같은 값 출력
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.3, random_state=11) 

# 분할된 데이터의 shape 확인
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

# 4. KNeighborsClassifier 의 객체 생성, k값은 8
knn = KNeighborsClassifier(n_neighbors=8)
print(type(knn))

# 5. 훈련 데이터를 이용하여 분류 모델 학습
knn.fit(x_train, y_train)

# 6. 학습된 knn 모델 기반 예측
y_pred = knn.predict(x_test)
print('Prediction:\n',y_pred)

# 7. 모델 평가, 인자는 test용 data set
score = knn.score(x_test, y_test)
print('Accuracy : {0:.5f}'.format(score))