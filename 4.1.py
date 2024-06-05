import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from skimage import feature
img = cv2.imread('timg.jpg', cv2.IMREAD_GRAYSCALE)
features = ft.hog(img,orientations=6,pixels_per_cell=[20,20],cells_per_block=[2,2],visualize=True)
plt.imshow(features[1],cmap=plt.cm.gray)
plt.show()

def extract_lbp_features(image):
# 将图像转换为灰度图像
gray_image = image.convert('L')
# 计算 LBP 特征
lbp = feature.local_binary_pattern(gray_image, 8, 1.0)
#HOG特征
hog_image = self.render_gradient(np.zeros([height, width]), cell_gradient_vector)


import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

img_path = r'./photo/2-5-1250-1.bmp'
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# 使用SIFT
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)
#PCA
descriptor = StandardScaler().fit_transform(descriptor)
pca = PCA(n_components=50)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#朴素贝叶斯
#加载数据
X, y = load_iris(return_X_y=True)
X, y = pd.DataFrame(X[:100]), pd.DataFrame(y[:100])
#训练集、测试集划分
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3)

model = NaiveBayes(X_train, y_train)
y_pre = model.predict(X_test)
print(accuracy_score(y_pre, y_test))

#KNN分类器
from sklearn.model_selection import train_test_split  # 引入train_test_split函数
from sklearn.neighbors import KNeighborsClassifier  # 引入KNN分类器
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
knn = KNeighborsClassifier()  # 调用KNN分类器
knn.fit(x_train, y_train)  # 训练KNN分类器
y_pred = knn.predict(x_test)
print(y_pred)
print(y_test)
print("准确率：", accuracy_score(y_pred, y_test))
