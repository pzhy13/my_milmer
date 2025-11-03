import cv2
import mediapipe as mp
from tqdm import tqdm
import os
import multiprocessing
import sys
# import time # 移除了未使用的导入
# import matplotlib.pyplot as plt # 移除了未使用的导入

# 注意：MediaPipe 模型初始化已移入 process_participant_data 函数内部，
# 以确保在多进程环境中，每个子进程都能拥有一个独立、干净的模型实例。

# --- 核心处理函数 (用于并行) ---
def process_participant_data(participant_id):
    # --- 重新在进程内部初始化 MediaPipe model ---
    mp_face_detection = mp.solutions.face_detection
    model = mp_face_detection.FaceDetection(   
            min_detection_confidence=0.7, 
            model_selection=0,
    )
    
    participant_str = f's{participant_id:02}'
    
    video_input_dir = f'./face_video/{participant_str}/' 
    output_dir = f'./faces/{participant_str}/'
    
    results_summary = []

    if not os.path.exists(video_input_dir):
        return f"警告 (s{participant_id:02}): 找不到视频目录 {video_input_dir}。"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = [f for f in os.listdir(video_input_dir) if f.lower().endswith('.avi')]
    
    if not video_files:
        return f"警告 (s{participant_id:02}): 在 {video_input_dir} 中没有找到 .avi 文件。"

    for video_file in video_files:
        video_path = os.path.join(video_input_dir, video_file)
        
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            results_summary.append(f"错误: 无法打开视频文件: {video_path}")
            continue
        
        # print(f"--- 正在处理视频: {video_path} ---") # 在多进程中，只打印摘要
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_filename_base = os.path.splitext(video_file)[0]
        
        frames_saved = 0
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.process(img_RGB)

            if results.detections:
                detection = results.detections[0] 
                
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                bbox = [int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)]
                
                expansion_ratio = 0.1
                bbox_expanded = [max(0, bbox[0] - bbox[2] * expansion_ratio),
                                 max(0, bbox[1] - bbox[3] * expansion_ratio),
                                 min(iw, bbox[0] + bbox[2] * (1 + expansion_ratio)),
                                 min(ih, bbox[1] + bbox[3] * (1 + expansion_ratio))]

                face_image = frame[int(bbox_expanded[1]):int(bbox_expanded[3]), int(bbox_expanded[0]):int(bbox_expanded[2])]
                
                output_filename = os.path.join(output_dir, f'{video_filename_base}_frame_{frame_count:05d}.png')
                
                if face_image.size > 0:
                    cv2.imwrite(output_filename, face_image)
                    frames_saved += 1

            frame_count += 1
        
        cap.release()
        results_summary.append(f"完成视频 {video_file} (总帧数: {total_frames}, 保存帧数: {frames_saved})")
    
    return f"被试 s{participant_id:02} 处理完毕. 详情: {'; '.join(results_summary)}"

# --- 主启动块 (并行) ---
if __name__ == '__main__':
    # 21 个被试，排除 s11
    ALL_SUBJECTS = [s for s in range(1, 23) if s != 11]
    
    # 使用 8 个核心进行并行处理
    num_processes = 8 
    
    print(f"--- 启动 {num_processes} 个进程并行处理 {len(ALL_SUBJECTS)} 个被试的人脸视频数据 ---")
    
    # 使用 Pool 来分配任务
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 并用 tqdm 包装结果迭代器
        results = list(tqdm(
            pool.imap_unordered(process_participant_data, ALL_SUBJECTS), 
            total=len(ALL_SUBJECTS), 
            desc="Face Detection Preprocessing (By Subject)"
        ))
    
    for result in results:
        if result and ("警告" in result or "错误" in result):
             print(result, file=sys.stderr)
        elif result:
             print(result)

    print("--- 所有人脸数据处理完毕 ---")