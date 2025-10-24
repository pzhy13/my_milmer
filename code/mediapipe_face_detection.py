import cv2
import mediapipe as mp
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection
model = mp_face_detection.FaceDetection(   
        min_detection_confidence=0.7, 
        model_selection=0,
)

def process_participant_data(participant_id):
    participant_str = f's{participant_id:02}'
    
    # *** MODIFICATION 1: 更改输入和输出目录 ***
    # 假设你的 DEAP 视频 位于
    # ./face_video/s01/s01_trial01.avi
    # ./face_video/s01/s01_trial02.avi
    # ...
    video_input_dir = f'./face_video/{participant_str}/' # 更改为你的 AVI 视频所在的文件夹
    output_dir = f'./faces/{participant_str}/'

    if not os.path.exists(video_input_dir):
        print(f"警告: 找不到视频目录 {video_input_dir}。跳过参与者 {participant_id}。")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # *** MODIFICATION 2: 循环遍历 .avi 视频文件 ***
    video_files = [f for f in os.listdir(video_input_dir) if f.lower().endswith('.avi')]
    
    if not video_files:
        print(f"警告: 在 {video_input_dir} 中没有找到 .avi 文件。")
        return

    for video_file in video_files:
        video_path = os.path.join(video_input_dir, video_file)
        
        # *** MODIFICATION 3: 使用 VideoCapture 打开视频 ***
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"错误: 无法打开视频文件: {video_path}")
            continue
        
        print(f"--- 正在处理视频: {video_path} ---")
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_filename_base = os.path.splitext(video_file)[0] # 例如 "s01_trial01"

        # 使用 tqdm 显示处理进度
        pbar = tqdm(total=total_frames, desc=f"处理 {video_file}")

        # *** MODIFICATION 4: 循环读取视频的每一帧 ***
        while cap.isOpened():
            ret, frame = cap.read()
            
            # 如果 'ret' 是 False, 说明视频结束了
            if not ret:
                pbar.close()
                break

            # 'frame' 变量现在等同于你原始代码中的 'img' 变量
            # --- 以下是你的原始人脸检测逻辑 ---
            
            img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.process(img_RGB)

            # *** MODIFICATION 5: 增加安全检查 ***
            # 你的原始代码假设总能检测到人脸 (results.detections[0])
            # 如果没有检测到人脸，它会崩溃。我们必须检查 'results.detections'
            if results.detections:
                detection = results.detections[0] # 假设每帧只有一张脸
                
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = [int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)]
                
                # 你的 10% 扩展逻辑 (无变化)
                expansion_ratio = 0.1
                bbox_expanded = [max(0, bbox[0] - bbox[2] * expansion_ratio),  # left
                                 max(0, bbox[1] - bbox[3] * expansion_ratio),  # top
                                 min(iw, bbox[0] + bbox[2] * (1 + expansion_ratio)), # right
                                 min(ih, bbox[1] + bbox[3] * (1 + expansion_ratio))] # bottom

                # 从原始 'frame' 中裁剪出人脸
                face_image = frame[int(bbox_expanded[1]):int(bbox_expanded[3]), int(bbox_expanded[0]):int(bbox_expanded[2])]
                
                # *** MODIFICATION 6: 保存被裁剪的人脸帧 ***
                # 我们需要一个唯一的文件名，例如:
                # s01_trial01_frame_00001.png
                # s01_trial01_frame_00002.png
                # ...
                output_filename = f'{output_dir}{video_filename_base}_frame_{frame_count:05d}.png'
                
                # 确保裁剪的图像不为空
                if face_image.size > 0:
                    cv2.imwrite(output_filename, face_image)
                else:
                    print(f"警告: 帧 {frame_count} 裁剪为空 ({video_file})")

            else:
                # （可选）处理未检测到人脸的帧
                # print(f"警告: 帧 {frame_count} 未检测到人脸 ({video_file})")
                pass

            frame_count += 1
            pbar.update(1)
        
        # 释放视频捕获对象
        cap.release()
        print(f"--- 完成: {video_file} ---")

for participant_id in range(1, 23):
    process_participant_data(participant_id)