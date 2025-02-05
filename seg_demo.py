from ultralytics import YOLO
import cv2
import numpy as np


class YoloSegmentation:
    def __init__(self, model_path: str, image_path: str):
        # 加载模型
        self.model = YOLO(model_path, task='segment')
        self.image_path = image_path
        self.results = None

    def predict(self):
        # 进行预测
        self.results = self.model.predict(self.image_path, save=True)

    def get_min_bounding_boxes(self):
        # 存储最小外接四边形的角点
        bounding_boxes = []

        # 遍历所有检测结果
        if self.results is not None:
            for result in self.results:
                # 检查是否有分割掩码
                if result.masks is not None:
                    # 遍历每个实例的掩码
                    for i, mask in enumerate(result.masks.xy):
                        # 将坐标转换为浮点型
                        points = mask.astype(np.float32)

                        # 计算最小外接旋转矩形
                        rotated_rect = cv2.minAreaRect(points)

                        # 获取旋转矩形的四个角点
                        box_points = cv2.boxPoints(rotated_rect)

                        # 转换为整数坐标
                        box_points = np.int32(box_points)

                        # 将角点存储到bounding_boxes列表
                        bounding_boxes.append(box_points)

        return bounding_boxes

    def draw_bounding_boxes(self):
        # 可视化：将最小外接四边形绘制到图像上
        if self.results is not None:
            for result in self.results:
                # 原始图像
                img = result.orig_img
                # 获取最小外接四边形的角点
                bounding_boxes = self.get_min_bounding_boxes()

                # 绘制每个四边形
                for box_points in bounding_boxes:
                    cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)

                # 显示图像
                cv2.imshow("Bounding Boxes", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


# 使用示例
model_path = "/Users/wenkang/PycharmProjects/SizeEstimation/src/ultralytics/model/yolo11n-seg.pt"
image_path = "demo.jpg"

yolo_segmenter = YoloSegmentation(model_path, image_path)
yolo_segmenter.predict()  # 执行预测
bounding_boxes = yolo_segmenter.get_min_bounding_boxes()
print(bounding_boxes)
yolo_segmenter.draw_bounding_boxes()  # 绘制最小外接四边形
