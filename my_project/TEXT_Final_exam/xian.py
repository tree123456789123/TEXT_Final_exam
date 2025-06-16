import os
import pandas as pd

# 图像目录和精简后的CSV文件
img_dir = 'data_image'
train_csv = 'reduced_train.csv'  # 精简后的训练集
test_csv = 'reduced_test.csv'  # 精简后的测试集

# 读取精简后的CSV
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# 获取所有需要保留的图像文件名
train_images = set(train_df['image'].str.replace("\\", "/"))
test_images = set(test_df['image'].str.replace("\\", "/"))

# 合并训练集和测试集的图像文件名
all_images_to_keep = train_images.union(test_images)

# 遍历图像目录，删除不在保留列表中的图像
for img_name in os.listdir(img_dir):
    if img_name not in all_images_to_keep:
        img_path = os.path.join(img_dir, img_name)
        if os.path.isfile(img_path):
            os.remove(img_path)  # 删除图像文件
            print(f"Deleted: {img_name}")

