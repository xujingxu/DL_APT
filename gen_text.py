import os
import glob



# 生成图像路径和标签
def generate_labels_and_paths(root_folder):
    labels_and_paths = []
    
    for root, dirs, files in os.walk(root_folder):
        # 遍历每个文件
        for file_name in files:
            if file_name.endswith('.png'):
                # 获取文件路径
                file_path = os.path.join(root, file_name)
                
                # 获取倒数第二层目录作为标签
                label = os.path.basename(os.path.dirname(root))
                
                # 添加标签和路径到列表
                labels_and_paths.append((file_path,label))
    
    return labels_and_paths

 # 定义根文件夹路径
root_folder_path = '/mnt/user/data/png/'

# 调用函数生成标签和路径列表
labels_and_paths = generate_labels_and_paths(root_folder_path)


# 将图像路径和标签保存到txt文件中
with open('/mnt/users/data/train.txt', 'w') as f:
    for image_path, label in labels_and_paths:
        f.write(f"{image_path} {label}\n")
