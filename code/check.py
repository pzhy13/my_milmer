import os
import glob

# --- 配置 ---
# 根据论文 ，排除 s11
SUBJECTS_TO_CHECK = [s for s in range(1, 23) if s != 11] # s01-s10, s12-s22
EEG_DIR = './EEGData'
FACE_DIR = './faces'
# --- (DEAP数据集中每个被试有40个trial) ---
TRIALS_PER_SUBJECT = 40 

print(f"--- Milmer 数据完整性检查 ---")
print(f"将检查 {len(SUBJECTS_TO_CHECK)} 个被试 (s01-s22, 排除 s11)...")
print(f"EEG 目录: {EEG_DIR}")
print(f"Faces 目录: {FACE_DIR}\n")

all_ok = True
missing_files = []

for subject_id in SUBJECTS_TO_CHECK:
    subject_str = f's{subject_id:02}'
    subject_ok = True
    
    print(f"--- 正在检查被试: {subject_str} ---")
    
    # 1. 检查 EEG 文件
    eeg_file = os.path.join(EEG_DIR, f'{subject_str}_eeg.npy')
    label_file = os.path.join(EEG_DIR, f'{subject_str}_labels.npy')
    
    if not os.path.exists(eeg_file):
        print(f"[失败] ❌: 找不到 {eeg_file}")
        missing_files.append(eeg_file)
        subject_ok = False
        
    if not os.path.exists(label_file):
        print(f"[失败] ❌: 找不到 {label_file}")
        missing_files.append(label_file)
        subject_ok = False
        
    if subject_ok:
        print(f"[通过] ✅: EEG 和 Labels 文件存在。")

    # 2. 检查 Face 目录
    subject_face_dir = os.path.join(FACE_DIR, subject_str)
    
    if not os.path.isdir(subject_face_dir):
        print(f"[失败] ❌: 找不到人脸目录 {subject_face_dir}")
        missing_files.append(subject_face_dir)
        subject_ok = False
    else:
        # 检查目录是否为空
        face_files = os.listdir(subject_face_dir)
        if len(face_files) == 0:
            print(f"[失败] ❌: 人脸目录 {subject_face_dir} 是空的！")
            missing_files.append(f"{subject_face_dir} (is empty)")
            subject_ok = False
        else:
            # 抽查第一个 trial
            trial_01_files = glob.glob(os.path.join(subject_face_dir, f'{subject_str}_trial01_frame_*.png'))
            if not trial_01_files:
                 print(f"[警告] ⚠️ : 在 {subject_face_dir} 中未找到 trial01 的文件。")
                 # 这不一定是致命错误，因为 train.py 会跳过，但最好检查一下
            
            print(f"[通过] ✅: 人脸目录存在且包含 {len(face_files)} 个文件。")

    if not subject_ok:
        all_ok = False

# --- 总结 ---
print("\n--- 检查完毕 ---")
if all_ok:
    print("✅✅✅ 恭喜！所有 21 个被试的数据都已准备就绪。")
    print("你可以开始运行 train.py 脚本了。")
else:
    print(f"❌❌❌ 检查失败！缺失以下 {len(missing_files)} 个关键文件或目录：")
    for f in missing_files:
        print(f" - {f}")
    print("请重新运行预处理脚本 (EEG_preprocessing.ipynb 和 mediapipe_face_detection.py) 来生成这些文件。")