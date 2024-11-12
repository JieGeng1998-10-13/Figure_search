import os
import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

w, h = 224, 224
# 加载模型
encoder = ResNet50(include_top=False)


#base_path = r"D:\datasets_test\datasets"
base_path = r"/mnt/workspace/datasets"
# 获取所有图片路径
files = [os.path.join(base_path, file) for file in os.listdir(base_path)]
# 将图片转换成向量
embeddings = []
for file in files:
    # 读取图片，转换成与目标图片同一尺寸
    source = cv2.imread(file)
    if not isinstance(source, np.ndarray):
        continue
    source = cv2.resize(source, (w, h))
    embedding = encoder(preprocess_input(source[None]))
    embeddings.append({
        "filepath": file,
        "embedding": tf.reshape(embedding, (-1,))
    })
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

