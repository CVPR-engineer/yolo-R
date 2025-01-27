# @File: test.py
# @Author: chen_song
# @Time: 2024/11/26 下午7:14
import argparse

import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box


def run_yolov5_detection(weights_path='best.pt', source='test_images/', img_size=640, conf_thres=0.25, iou_thres=0.45):
    # 加载模型
    global result, label
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型文件
    model = attempt_load(weights_path, map_location=device)  # 加载权重文件
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    model.to(device).eval()

    # 加载数据源（这里以加载图像为例，也可以修改为加载视频等情况）===最后每个图像作为一个个对象被加载进去
    dataset = LoadImages(source, img_size=img_size, stride=stride)

    # 开始检测===这里指的是各种格式数据集
    for path, img, im0s, vid_cap,_ in dataset:
    # for img in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        with torch.no_grad():
            pred = model(img)[0]

        # 应用非极大值抑制（NMS）来筛选检测结果
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        # 处理检测结果并绘制
        for i, det in enumerate(pred):
            if det is not None:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    result = plot_one_box(xyxy, im0s, label,color=(0, 255, 0), line_thickness=3)
                    cv2.imwrite(r'C:\Users\PC\Desktop\output\output.jpg', result)
                    # cv2.imshow('output', result)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
        print(result,label)
        return (result,label)

        # 这里可以根据实际需求进一步处理绘制后的图像，比如保存图像等
        # 例如：cv2.imwrite('output_image.jpg', im0s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    weight_path = r'C:\Users\PC\Desktop\test\best.pt'
    source = r'E:\paperCode\yolov5-master\output.jpg'
    img_size = 640
    conf_thres = 0.7
    iou_thres = 0.6
    run_yolov5_detection(weight_path, source, img_size, conf_thres, iou_thres)
