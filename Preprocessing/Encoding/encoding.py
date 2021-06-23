# 필요한 라이브러리 로딩
import seaborn as sns

# sklearn 의 LabelEncoder, OneHotEncoder 를 import 시키기
from sklearn.preprocessing import LabelEncoder 
from  sklearn.preprocessing import OneHotEncoder

# seaborn에 내당된 tips 데이터셋 로딩
tips = sns.load_dataset('tips')

#저장된 data를 5개의 행으로 출력
print(tips.head(5))

#print('Tips Categoris: \n'tips['day'].unique()) 카테고리 확인

# 인코딩할 컬럼 데이터 준비 
items = tips['day']

# 1. 라벨인코딩(LabelEncoding) 실습
# LabelEncoder 객체 생성
encoder = LabelEncoder()

# fit 메소드에 인코딩할 데이터 전달
encoder.fit(items)

# transform 메소드를 통해 데이터 변환 
labels = encoder.transform(items)

# 인코딩 결과 출력 (코드 제출시에는 주석 처리)
#print('Label Encoding Result:\n',labels)

# 인코딩된 수치형 데이터의 실제 클래스 확인 및 출력
classes = encoder.classes_
print('LabelEncoding classes:', classes)

# 디코딩 결과 확인 및 출력
inverse_result = encoder.inverse_transform([2])
print('LabelDecoding result:', inverse_result)

# 2. 원핫인코딩(OneHotEncoding) 실습
# 원핫인코더의 입력 데이터는 2차원 데이터만 가능, 변환 필요(reshape)
labels = labels.reshape(-1,1)

# OneHotEncoder 객체 생성
one_hot_encoder = OneHotEncoder()

# .fit 메소드에 인코딩할 데이터 전달
one_hot_encoder.fit(labels)

# .transform 메소드를 통해 데이터 변환 
one_hot_labels = one_hot_encoder.transform(labels)

# 인코딩 결과 출력, toarray()는 배열 형태로 출력
#print('OneHotEncoding Result:\n', one_hot_labels.toarray())

# categories_ 속성을 통해 인코딩 데이터의 실제 클래스 확인 가능
onehot_classes = one_hot_encoder.categories_
print('OneHotEncoding classes:', onehot_classes)
