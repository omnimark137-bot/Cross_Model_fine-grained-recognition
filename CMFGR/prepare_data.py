import os
import glob
import shutil
import random
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你的原始数据根目录 (请修改这里！)
# 注意：路径中不要包含中文字符，如果必须包含，请确保编码正确
SOURCE_ROOT = r'D:\paired_dataset\REID'

# 2. 目标数据保存目录 (通常是项目下的 data/AircraftShip)
TARGET_ROOT = r'./data/AircraftShip'

# 3. 数据集划分比例
# 建议：60% 用于训练，10% 用于 Query (查询)，30% 用于 Gallery (被查询库)
TRAIN_RATIO = 0.6
QUERY_RATIO = 0.1


# GALLERY_RATIO = 1 - 0.6 - 0.1 = 0.3

# ===========================================

def prepare_data():
    # 1. 创建目标目录结构
    train_dir = os.path.join(TARGET_ROOT, 'bounding_box_train')
    query_dir = os.path.join(TARGET_ROOT, 'query')
    gallery_dir = os.path.join(TARGET_ROOT, 'bounding_box_test')

    for d in [train_dir, query_dir, gallery_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # 2. 获取所有子文件夹 (例如 A220_RGB, A220_SAR ...)
    # 假设源目录下都是 类别_模态 格式的文件夹
    sub_dirs = glob.glob(os.path.join(SOURCE_ROOT, '*_*'))

    print(f"Found {len(sub_dirs)} sub-directories in source root.")

    total_files = 0

    # 3. 遍历每个文件夹进行处理
    for sub_dir in sub_dirs:
        folder_name = os.path.basename(sub_dir)

        # 跳过非数据文件夹
        if not os.path.isdir(sub_dir):
            continue

        # 获取该文件夹下所有 .tif 图片
        img_paths = glob.glob(os.path.join(sub_dir, '*.tif'))
        if len(img_paths) == 0:
            print(f"No .tif images found in {folder_name}, skipping.")
            continue

        print(f"Processing {folder_name}: {len(img_paths)} images...")

        # 打乱顺序，实现随机划分
        random.shuffle(img_paths)

        # 计算切分点
        num_imgs = len(img_paths)
        num_train = int(num_imgs * TRAIN_RATIO)
        num_query = int(num_imgs * QUERY_RATIO)

        # 划分列表
        train_imgs = img_paths[:num_train]
        query_imgs = img_paths[num_train: num_train + num_query]
        gallery_imgs = img_paths[num_train + num_query:]

        # 定义复制函数
        def copy_files(file_list, target_dir):
            for src_path in file_list:
                filename = os.path.basename(src_path)
                dst_path = os.path.join(target_dir, filename)
                shutil.copy2(src_path, dst_path)  # 使用 copy2 保留文件元数据

        # 执行复制
        copy_files(train_imgs, train_dir)
        copy_files(query_imgs, query_dir)
        copy_files(gallery_imgs, gallery_dir)

        total_files += num_imgs

    print(f"\nData preparation finished! Total {total_files} images processed.")
    print(f"Data saved to: {TARGET_ROOT}")


if __name__ == '__main__':
    prepare_data()