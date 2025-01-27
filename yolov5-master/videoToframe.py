"""
@Time    : 2024/10/2 上午9:46
@Author  : ChenSong
@File    : videoToframe.py
@Desc    : 
"""

# 在此处添加你的代码
import cv2
import os

videos_src_path = r"E:\paperCode\yolov5-master\runs\detect\exp44"  # 视频文件夹路径
videos_save_path = r"C:\Users\PC\Desktop\frame"  # 保存图片的路径
prefix = 0
for video_name in os.listdir(videos_src_path):

    if video_name.endswith('.mp4'):  # 确保是视频文件
        video_path = os.path.join(videos_src_path, video_name)
        cap = cv2.VideoCapture(video_path)
        t = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if cap.isOpened():
            frame_id = 0
            while cap.read():  # 逐帧读取
                ret, frame = cap.read()
                if ret:
                    img_name = f"{prefix}-{frame_id}.jpg"
                    img_path = os.path.join(videos_save_path, img_name)
                    cv2.imwrite(img_path, frame)  # 保存图片
                    print(f'第{frame_id}张图片已保存'.format(frame_id))
                    frame_id += 1
                else:
                    break
        prefix = prefix+1
        cap.release()

