# 하이퍼파라미터 튜닝을 을 위한 GridSearchCV 라이브러리 로딩
from sklearn.model_selection import GridSearchCV

# 모델 구현을 위한 라이브러리 로딩
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# load_iris() 메서드를 이용하여 iris 데이터 셋 로드
iris = load_iris()

# 학습, 테스트 데이터셋 분리
x_train, x_test, y_train, y_test = train_test_split(iris.data, 
                                                    iris.target, 
                                                    test_size=0.2,
                                                    random_state=121)


# DecisionTreeClassifier 모델 객체 생성
dtree = DecisionTreeClassifier()

# 모델의 후보 파라미터 셋(param_grid)을 지정한 딕셔너리 객체 생성
parameters = {'max_depth':[1,2,3,], 'min_samples_split':[2,3]}

# GridSearchCV 객체 생성
grid_dtree = GridSearchCV(estimator=dtree, param_grid=parameters, cv=3, refit=True)

# GridSearchCV 객체의 fit() 메서드를 이용하여
# 후보 파라미터 셋의 성능 검증
grid_dtree.fit(x_train,y_train)

# 후보 파라미터 셋의 성능 검증 결과 출력
print('Optimal parameter:', grid_dtree.best_params_)
print('Max accuracy: {0:.4f}'.format(grid_dtree.best_score_))

# 최적의 파라미터 모델을 이용하여 예측값 생성
estimator = grid_dtree.best_estimator_
pred = estimator.predict(x_test)

# 최적의 파라미터 모델의 성능지표 출력
print('Test accuracy: {0:.4f}'.format(accuracy_score(y_test,pred))) 
