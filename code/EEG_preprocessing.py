import scipy.io
import numpy as np
import pandas as pd
import mne
from mne import io
from mne.preprocessing import ICA
import os

# 确保保存数据的目录存在
output_dir = './EEGData'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"创建目录: {output_dir}")

# --- MNE 设置 (来自你原来的第5单元格) ---
sfreq = 128
channels = 32
samples = 384  # 3 seconds * 128Hz
num_each_trial = 20 # 60s / 3s = 20 segments
ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 
            'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 
            'FC6', 'FC2', 'F4', 'F8', 'AF4', 'FP2', 'Fz', 'Cz']
ch_types = ['eeg'] * channels

# --- 循环处理 1 到 22 号被试 ---
for participant_id in range(1, 23):
    subject_str = f's{participant_id:02}'
    mat_file_path = f'./data_preprocessed_matlab/{subject_str}.mat'
    
    # 检查 .mat 文件是否存在
    if not os.path.exists(mat_file_path):
        print(f"警告: 找不到文件 {mat_file_path}，跳过被试 {subject_str}")
        continue
        
    print(f"--- 正在处理被试: {subject_str} ---")

    # --- 1. 加载数据 (来自你原来的第2单元格) ---
    mat_data = scipy.io.loadmat(mat_file_path) 
    original_data = mat_data['data']
    original_label = mat_data['labels']

    # --- 2. 切片 (来自你原来的第3单元格) ---
    # original_data 形状为 (40, 40, 8064)
    # 40个trial, 40个通道 (前32个是EEG), 8064个采样点
    # 我们丢弃前3秒 (3*128=384个点) 的基线
    sliced_data = original_data[:, :32, 384:] 
    # print(f"{subject_str} sliced_data: {sliced_data.shape}") # (40, 32, 7680)
    eeg_data = sliced_data

    # --- 3. 处理标签 (来自你原来的第4单元格) ---
    valence = original_label[:,0]
    arousal = original_label[:,1]
    # 1(HAHV), 2(LAHV), 3(LALV), 4(HALV) -> 映射为 0, 1, 2, 3
    VA_labels = np.where((valence > 5) & (arousal > 5), 0,
              np.where((valence >= 5) & (arousal < 5), 1,
                np.where((valence < 5) & (arousal < 5), 2, 3)))
    
    segment_size = 3 * 128 # 384
    num_segments = sliced_data.shape[2] // segment_size # 7680 / 384 = 20
    
    # 将每个trial的1个标签扩展为20个segment的标签
    expanded_VA_labels = np.repeat(VA_labels, num_segments) # (40*20 = 800,)
    labels = expanded_VA_labels 

    # --- 4. MNE 预处理 (来自你原来的第5单元格) ---
    data_list = []
    # (40, 32, 7680) -> 列表，包含40个 (1, 32, 7680) 的数组
    eeg_data_segments = np.split(eeg_data, 40, axis=0) 
    
    for segment in eeg_data_segments:
        # segment 形状 (1, 32, 7680), 重塑为 (32, 7680)
        segment_2d = segment.reshape(-1, channels).T

        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = mne.io.RawArray(segment_2d, info=info, verbose=False)

        # 滤波
        raw.filter(l_freq=1.0, h_freq=50.0, verbose=False)

        # ICA
        ica = ICA(n_components=channels, random_state=0, max_iter=1000, verbose=False) 
        ica.fit(raw, verbose=False)
        ica.exclude = [] 
        ica.apply(raw, verbose=False)

        # (32, 7680) -> (7680, 32)
        data = raw.get_data().T 

        # (7680, 32) -> (20, 384, 32)
        # 确保数据正好是 20 * 384
        data = data[:num_each_trial * samples, :]
        data = data.reshape(num_each_trial, samples, channels)

        data_list.append(data)

    # 40个 (20, 384, 32) 数组 -> (800, 384, 32)
    data_array = np.concatenate(data_list, axis=0) 
    # (800, 384, 32) -> (800, 32, 384) 以匹配模型输入
    data_array = np.swapaxes(data_array, 1, 2) 
    eeg_data_processed = data_array

    # --- 5. 保存 .npy 文件 (来自你原来的第6单元格) ---
    eeg_save_path = os.path.join(output_dir, f'{subject_str}_eeg.npy')
    labels_save_path = os.path.join(output_dir, f'{subject_str}_labels.npy')
    
    np.save(eeg_save_path, eeg_data_processed) # (800, 32, 384)
    np.save(labels_save_path, labels) # (800,)
    
    print(f"成功保存: {eeg_save_path} 和 {labels_save_path}")

print("--- 所有被试处理完毕 ---")