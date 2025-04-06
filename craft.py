import os

def rename_subfolders(root_dir):
    # 获取指定目录下的所有子文件夹
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    # 遍历子文件夹并重命名
    for index, folder in enumerate(subfolders, start=1):
        old_folder_path = os.path.join(root_dir, folder)
        new_folder_name = f'class{index}'
        new_folder_path = os.path.join(root_dir, new_folder_name)

        # 重命名文件夹
        os.rename(old_folder_path, new_folder_path)
        print(f'Renamed: {old_folder_path} -> {new_folder_path}')

# 使用示例：指定根目录
root_directory = '/root/workspace/nmode_cifar10/ml-cvnets/dataset_r/validation'
rename_subfolders(root_directory)
