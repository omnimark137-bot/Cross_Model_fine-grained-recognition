# 集成PGD的YOLO模型主文件
# 可根据需要从PGD/models/PGD_Lite_yolo.py迁移核心类和方法


from .PGD_Lite_yolo import Detect

# 可根据需要封装接口
class PGD_YOLO_Wrapper:
    def __init__(self, *args, **kwargs):
        self.model = Detect(*args, **kwargs)
    def forward(self, x):
        return self.model(x)
