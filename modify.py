import os
import shutil

val_folder = r'D:\Study\Deep_Learnig\Code_deep\tiny-imagenet-200\val'  # val文件夹路径
annotations_file = r'D:\Study\Deep_Learnig\Code_deep\tiny-imagenet-200\val\val_annotations.txt'  # 替换为val_annotations.txt文件路径

# 创建类别文件夹
with open(annotations_file, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        image_name, category = line[0], line[1]
        category_folder = os.path.join(val_folder, category)
        os.makedirs(category_folder, exist_ok=True)

# 将图像移动到相应的类别文件夹中
image_folder = os.path.join(val_folder, 'images')
with open(annotations_file, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        image_name, category = line[0], line[1]
        src_path = os.path.join(image_folder, image_name)
        dest_folder = os.path.join(val_folder, category)
        dest_path = os.path.join(dest_folder, image_name)
        shutil.move(src_path, dest_path)

# 删除原始的images文件夹
shutil.rmtree(image_folder)