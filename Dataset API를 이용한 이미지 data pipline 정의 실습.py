import tensorflow as tf
from tensorflow import keras
import PIL
import PIL.Image
import numpy as np
import os

np.random.seed(42)

# 1. 이미지 입력 파이프라인 만들기
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
            fname='flower_photos')

# pathlib으로 glob을 이용해서 특정 패턴 검색
image_count = len(list(data_dir.glob('*/*.jpg')))

# 모든 파일의 모든 데이터 가져오기
file_paths = [str(glob_path) for glob_path in data_dir.glob('*/*.jpg')]

file_pattern = str(data_dir/'*/*')
dataset = tf.data.Dataset.list_files(file_pattern=file_pattern, shuffle=True)

# 파일 트리 구조 사용하여 class_names 목록 컴파일
class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

# 데이터셋 분할
val_size = int(image_count * 0.2)
train_set = dataset.skip(val_size)
validation_set = dataset.take(val_size)

batch_size = 328
img_height = 180
img_width = 180

# broadcasting을 통해 boolean Index를 얻어서 라벨 인코딩하기
def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    ont_hot = parts[-2] == class_names
    return tf.argmax(one_hot)

# 이미지 읽어들이는 라이브러리
img = tf.io.read_file('/content/datasets/flower_photos/roses/4918137796_21f0922b0c_n.jpg')

# img에서 얻은 binary 파일을 실제 텐서 객체로 변환
img = tf.image.decode_jpeg(img)

def decode_img(img):
  # 이미지들을 file_read를 해서 텐서 객체로 변환
  img = tf.image.decode_jpeg(img)
  return tf.image.resize(images = img, size=(img_height,img_width))

def process_path(file_path):
  label = get_label(file_path)
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Dataset.map 사용해서 image, label 쌍의 데이터셋 작성

# autotune = -1, tf.experimental
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_set = train_set.map(process_path, num_parallel_calls=AUTOTUNE)
validation_set = validation_set.map(process_path, num_parallel_calls=AUTOTUNE)

# ds.cache는 이미지 파일을 텐서객체로 바꾼 후, 실제 모델한테 학습을 시킴
def configure_for_performance(ds):
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=1000)
  ds = ds.batch(batch_size=batch_size)

  # AUTOTUNE으로 몇개를 미리 가져올건지 동적으로 값 조정
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds