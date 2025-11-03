import scipy.io
import numpy as np
import pandas as pd
import mne
from mne import io
from mne.preprocessing import ICA
import os
import multiprocessing
import sys 
from tqdm import tqdm # 用于显示总进度

# 确保保存数据的目录存在
output_dir = './EEGData'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- MNE 设置 (全局常量) ---
sfreq = 128
channels = 32
samples = 384  # 3 seconds * 128Hz
num_each_trial = 20 # 60s / 3s = 20 segments
ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 
            'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 
            'FC6', 'FC2', 'F4', 'F8', 'AF4', 'FP2', 'Fz', 'Cz']
ch_types = ['eeg'] * channels

# --- 核心处理函数 (用于并行) ---
def process_single_subject(participant_id):
    subject_str = f's{participant_id:02}'
    mat_file_path = f'./data_preprocessed_matlab/{subject_str}.mat'
    
    # 检查 .mat 文件是否存在
    if not os.path.exists(mat_file_path):
        return f"警告 (s{participant_id:02}): 找不到文件 {mat_file_path}。"
        
    try:
        # --- 1. 加载数据 ---
        mat_data = scipy.io.loadmat(mat_file_path) 
        original_data = mat_data['data']
        original_label = mat_data['labels']

        # --- 2. 切片 ---
        sliced_data = original_data[:, :32, 384:] 
        eeg_data = sliced_data

        # --- 3. 处理标签 ---
        valence = original_label[:,0]
        arousal = original_label[:,1]
        VA_labels = np.where((valence > 5) & (arousal > 5), 0,
                  np.where((valence >= 5) & (arousal < 5), 1,
                    np.where((valence < 5) & (arousal < 5), 2, 3)))
        
        segment_size = 3 * 128 # 384
        num_segments = sliced_data.shape[2] // segment_size 
        
        expanded_VA_labels = np.repeat(VA_labels, num_segments) 
        labels = expanded_VA_labels 

        # --- 4. MNE 预处理 ---
        data_list = []
        eeg_data_segments = np.split(eeg_data, 40, axis=0) 
        
        for segment in eeg_data_segments:
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

            data = raw.get_data().T 

            # 重塑数据
            data = data[:num_each_trial * samples, :]
            data = data.reshape(num_each_trial, samples, channels)

            data_list.append(data)

        data_array = np.concatenate(data_list, axis=0) 
        data_array = np.swapaxes(data_array, 1, 2) 
        eeg_data_processed = data_array

        # --- 5. 保存 .npy 文件 ---
        eeg_save_path = os.path.join(output_dir, f'{subject_str}_eeg.npy')
        labels_save_path = os.path.join(output_dir, f'{subject_str}_labels.npy')
        
        np.save(eeg_save_path, eeg_data_processed)
        np.save(labels_save_path, labels)
        
        return f"成功保存: {eeg_save_path} 和 {labels_save_path}"

    except Exception as e:
        return f"错误 (s{participant_id:02})：处理时发生异常: {e}"


if __name__ == '__main__':
    # 21 个被试，排除 s11
    ALL_SUBJECTS = [s for s in range(1, 23) if s != 11]
    
    # 使用 8 个核心进行并行处理
    num_processes = 8 
    
    print(f"--- 启动 {num_processes} 个进程并行处理 {len(ALL_SUBJECTS)} 个被试的 EEG 数据 ---")
    
    # 使用 Pool 来分配任务
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 imap 并在主进程中用 tqdm 包装结果迭代器
        results = list(tqdm(
            pool.imap(process_single_subject, ALL_SUBJECTS), 
            total=len(ALL_SUBJECTS), 
            desc="EEG Preprocessing"
        ))
    
    for result in results:
        if result and ("警告" in result or "错误" in result):
             print(result, file=sys.stderr)
        elif result:
             print(result)

    print("--- 所有 EEG 被试数据处理完毕 ---")