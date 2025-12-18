# datasets/aircraft_ship.py
import glob
import os.path as osp
from .bases import BaseImageDataset


class AircraftShip(BaseImageDataset):
    """
    自定义数据集：包含飞机与船只
    文件名格式: 类别_序列号_模态.tif
    例如: A220_1_RGB.tif, Boeing737_51_SAR.tif
    """
    dataset_dir = 'AircraftShip'  # 数据集根目录下的文件夹名

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(AircraftShip, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # 1. 预扫描所有文件夹建立 类别名 -> 数字ID 的映射，确保训练集和测试集ID一致
        self.class_to_idx = self._generate_class_map()

        self.pid_begin = pid_begin

        # 2. 加载数据
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Aircraft & Ship Dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        # 为了兼容 HOSS 代码，我们需要计算一些统计信息
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

        # HOSS原本有Pair加载逻辑，如果你没有严格成对的数据（即每张RGB必有一张完全对应的SAR），可以置空
        # 如果需要成对训练，需要像 hoss.py 那样写 _process_dir_train_pair
        self.train_pair = []
        self.num_train_pair_pids = 0
        self.num_train_pair_imgs = 0
        self.num_train_pair_cams = 0

    def _generate_class_map(self):
        """扫描所有目录，获取所有唯一的类别名（如 A220, Boeing737）"""
        class_names = set()
        for d in [self.train_dir, self.query_dir, self.gallery_dir]:
            paths = glob.glob(osp.join(d, '*.tif'))
            for path in paths:
                filename = osp.basename(path)
                # 解析文件名: A220_1_RGB.tif -> A220
                class_name = filename.split('_')[0]
                class_names.add(class_name)

        # 排序保证顺序一致
        return {name: i for i, name in enumerate(sorted(list(class_names)))}

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.tif'))
        dataset = []

        for img_path in sorted(img_paths):
            filename = osp.basename(img_path)
            parts = filename.split('_')

            # 解析: "A220"_1_"RGB".tif
            class_name = parts[0]
            # sequence_id = parts[1] # 如果需要区分实例，可以用这个，但在细粒度分类中，我们通常把类别当做ID
            modality_str = parts[2].split('.')[0]  # RGB 或 SAR

            # 映射类别到数字 ID
            pid = self.class_to_idx[class_name]

            # 映射模态到 CamID (HOSS代码约定: RGB=0, SAR=1)
            if 'RGB' in modality_str:
                camid = 0
            elif 'SAR' in modality_str:
                camid = 1
            else:
                print(f"Warning: Unknown modality in {filename}, skipping.")
                continue

            if relabel:
                pid = self.pid_begin + pid

            # (path, pid, camid, trackid) -> HOSS 代码通常需要4个元素
            dataset.append((img_path, pid, camid, 1))

        return dataset

    def _check_before_run(self):
        """Check if all files are available"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))