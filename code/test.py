import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import SwinModel, AutoImageProcessor
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import logging
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict 

# --- 1. 导入已修正的 model.py ---
from model import MultiModalClassifier 

# --- 2. 定义全局 device (通用 cuda) ---
# (不再强制 cuda:0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# --- 3. 定义超参数 ---
ALL_SUBJECTS = [s for s in range(1, 23) if s != 11] 
NUM_CLASSES = 4 
NUM_INSTANCES = 10 
NUM_SELECT = 3     
MODEL_PATH = "./local_swin_model/" 
CHECKPOINT_PATH = './best_model_fast_run.pth' 

# --- 关键修改：设置 BATCH_SIZE (占位符) ---
# 警告：您必须使用 find_max_batch_size.py 找到最大值！
# 这里的 '12' 假设 3 块 GPU 每块跑 4 个。
BATCH_SIZE = 12 

# --- 4. 设置日志记录 (Logging) ---
def setup_logging(log_file='test_multi_gpu.log'): # 新日志名
    logging.getLogger().handlers = [] 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s - RANK_{rank} - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logging.info("--- 测试脚本日志系统已启动 (多 GPU, BS={BATCH_SIZE}) ---")

# --- 5. DEAP 数据集类 (无修改) ---
class DEAPDataset(Dataset):
    # ... (此处省略，与原文件相同) ...
    def __init__(self, subject_list, eeg_dir, face_dir, processor, num_instances=10):
        self.eeg_data = []
        self.labels = []
        self.image_bags = [] 
        self.sample_info = [] 
        self.processor = processor
        self.num_instances = num_instances
        logging.info(f"开始为 {len(subject_list)} 个被试加载数据...")
        for subject_id in tqdm(subject_list): # subject_id 是 1, 2, ... 10, 12, ...
            subject_str = f's{subject_id:02}'
            eeg_file = os.path.join(eeg_dir, f'{subject_str}_eeg.npy')
            label_file = os.path.join(eeg_dir, f'{subject_str}_labels.npy')
            if not (os.path.exists(eeg_file) and os.path.exists(label_file)):
                logging.warning(f"找不到被试 {subject_str} 的数据，已跳过。")
                continue
            subject_eeg_data = np.load(eeg_file) # 形状 (800, 32, 384)
            subject_labels = np.load(label_file)   # 形状 (800,)
            for sample_idx_in_subject in range(subject_eeg_data.shape[0]):
                trial_idx = sample_idx_in_subject // 20 # 结果是 0 到 39
                segment_idx = sample_idx_in_subject % 20  # 结果是 0 到 19
                self.eeg_data.append(subject_eeg_data[sample_idx_in_subject])
                self.labels.append(subject_labels[sample_idx_in_subject])
                self.sample_info.append((subject_id, trial_idx, segment_idx))
            current_bag_index = len(self.image_bags) # 记录添加 bag 前的长度
            expected_bags_for_subject = 40 * 20 # 预期每个被试添加 800 个 bag
            for trial_idx in range(40): # 40 trials
                trial_str = f'{subject_str}_trial{trial_idx+1:02}'
                frame_files = sorted(glob.glob(os.path.join(face_dir, subject_str, f'{trial_str}_frame_*.png')))
                if not frame_files:
                    for _ in range(20): 
                        self.image_bags.append([]) # 添加空列表
                    continue
                total_frames = len(frame_files)
                frames_per_segment_actual = total_frames // 20 
                if frames_per_segment_actual == 0: 
                    frames_per_segment_actual = total_frames
                for segment_idx in range(20): # 20 segments
                    start_frame_idx = segment_idx * frames_per_segment_actual
                    end_frame_idx = (segment_idx + 1) * frames_per_segment_actual
                    if segment_idx == 19: 
                        end_frame_idx = total_frames
                    segment_frame_files = frame_files[start_frame_idx:end_frame_idx]
                    if not segment_frame_files:
                        self.image_bags.append([frame_files[0]] if frame_files else []) # 使用第一帧或空列表
                    else:
                      self.image_bags.append(segment_frame_files) # 添加找到的帧列表
            added_bags = len(self.image_bags) - current_bag_index
            if added_bags != expected_bags_for_subject:
                 logging.error(f"被试 {subject_str} 的 image_bags 数量错误！预期 {expected_bags_for_subject}, 实际添加 {added_bags}")
        self.eeg_data = np.stack(self.eeg_data, axis=0)
        self.labels = np.array(self.labels)
        logging.info(f"总共加载 {len(self.labels)} 个样本。")
        assert len(self.eeg_data) == len(self.labels) == len(self.image_bags) == len(self.sample_info)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        eeg_segment = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        image_bag_files = self.image_bags[idx]
        subject_id, trial_idx, segment_idx = self.sample_info[idx]
        num_frames_in_bag = len(image_bag_files)
        if num_frames_in_bag == 0:
            image_bag_np = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_instances
        else:
            indices = np.linspace(0, num_frames_in_bag - 1, self.num_instances, dtype=int)
            image_bag_np = []
            for img_idx in indices:
                img_path = image_bag_files[img_idx]
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_resized = img.resize((224, 224))
                    img_np = np.array(img_resized)
                    if img_np.shape != (224, 224, 3):
                        img_np = np.zeros((224, 224, 3), dtype=np.uint8)
                    image_bag_np.append(img_np)
                except Exception as e:
                    logging.error(f"加载图像 {img_path} 失败: {e}. (来源: s{subject_id:02}, T{trial_idx + 1}, S{segment_idx}, F{img_idx}) 将使用空白图像。")
                    image_bag_np.append(np.zeros((224, 224, 3), dtype=np.uint8))
        images_tensor = torch.tensor(np.stack(image_bag_np), dtype=torch.uint8) 
        return eeg_segment, images_tensor, label


# --- 6. 评估函数 (无修改) ---
def evaluate(model, dataloader, device, mode='multimodal'):
    model.eval() 
    all_preds = []
    all_labels = []

    logging.info(f"评估将在设备 {device} (及 DataParallel 子卡) 上运行。")
    logging.info(f"开始在模式 '{mode}' 下评估...")
    eval_loop = tqdm(dataloader, desc=f"Testing ({mode})", leave=False)
    
    with torch.no_grad(): 
        for eeg_data, images_data, labels in eval_loop:
            # DataParallel 会自动将数据分发
            if mode == 'multimodal':
                eeg_input = eeg_data.to(device)
                image_input = images_data.to(device)
            elif mode == 'eeg_only':
                eeg_input = eeg_data.to(device)
                image_input = None 
            elif mode == 'image_only':
                eeg_input = None 
                image_input = images_data.to(device)
            else:
                raise ValueError("无效的评估模式")

            outputs = model(eeg_data=eeg_input, images_data=image_input)
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro') 
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, f1, cm

# --- 7. 主函数 (修改为 CPU -> DP -> GPU 逻辑) ---
def main():
    setup_logging(f'test_multi_gpu_bs{BATCH_SIZE}.log') 
    
    logging.info(f"将使用设备: {device}")
    if device.type == 'cpu':
         logging.warning("未检测到 CUDA 设备，将在 CPU 上运行。")

    # --- 关键修改：Swin 严格加载到 CPU ---
    logging.info("正在从本地加载 Swin Transformer (到 CPU)...")
    try:
        swin_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        swin_model = SwinModel.from_pretrained(MODEL_PATH) # 不加 .to(device)
        logging.info("Swin Transformer 加载完毕 (在 CPU)。")
    except Exception as e:
        logging.error(f"加载 Swin Transformer 失败: {e}")
        return

    # --- 关键修改：模型初始化严格在 CPU ---
    logging.info("初始化模型架构 (在 CPU)...")
    try:
        model_on_cpu = MultiModalClassifier(
            swin_processor=swin_processor,
            swin_model=swin_model, # 传递 CPU 上的 Swin
            # device=device, # 移除 device 参数
            num_classes=NUM_CLASSES,
            num_select=NUM_SELECT,
            num_instances=NUM_INSTANCES
        ) # 不加 .to(device)
    except Exception as e:
         logging.error(f"初始化 MultiModalClassifier 失败: {e}")
         return
         
    # --- 关键修改：加载权重到 CPU 模型 ---
    logging.info(f"正在从 {CHECKPOINT_PATH} 加载权重 (到 CPU)...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu') 
        new_state_dict = OrderedDict()
        is_data_parallel = any(key.startswith('module.') for key in checkpoint.keys())
        
        state_dict_to_load = checkpoint 
        if is_data_parallel:
            logging.info("检测到 DataParallel 权重，移除 'module.' 前缀...")
            for k, v in checkpoint.items():
                name = k[7:] 
                new_state_dict[name] = v
            state_dict_to_load = new_state_dict 
        
        # 将 state_dict 加载到 CPU 上的模型中
        model_on_cpu.load_state_dict(state_dict_to_load)
        logging.info("模型权重加载成功 (在 CPU)。")
    except Exception as e:
         logging.error(f"加载模型权重失败: {e}")
         logging.exception("加载 state_dict 时出错:") 
         return
         
    logging.info(f"将使用总 Batch Size = {BATCH_SIZE}")
    FINAL_BATCH_SIZE = BATCH_SIZE 
    
    logging.info("初始化完整数据集...")
    eeg_dir = './EEGData'
    face_dir = './faces'
    full_dataset = DEAPDataset(ALL_SUBJECTS, eeg_dir, face_dir, swin_processor, NUM_INSTANCES)
    
    RANDOM_SEED = 42 
    logging.info(f"正在使用种子 {RANDOM_SEED} 重新创建 80/20 训练/验证集划分...")
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2) 
    train_size = total_size - val_size 
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    logging.info(f"Train samples (未使用): {len(train_dataset)}, Val samples (用于测试): {len(val_dataset)}")
    
    logging.info(f"将使用全部 {len(val_dataset)} 个验证样本进行测试 (Batch Size = {FINAL_BATCH_SIZE})...")
    test_loader = DataLoader(val_dataset, batch_size=FINAL_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True if device.type=='cuda' else False)

    # --- 关键修改：先包装 DP，再 .to(device) ---
    model_to_test = model_on_cpu
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
        logging.info(f"检测到 {torch.cuda.device_count()} 个 GPU (可见: {visible_gpus})，应用 nn.DataParallel...")
        model_to_test = nn.DataParallel(model_to_test)
    elif device.type == 'cuda':
        logging.info("检测到 1 个 GPU。")

    logging.info("正在将模型移动到 GPU...")
    model_to_test.to(device)
    logging.info("模型已在 GPU 上。")
    # --- 修改结束 ---
    
    model_to_test.eval() # 确保设为评估模式

    # --- 执行评估 ---
    results = {}
    for mode in ['multimodal', 'eeg_only', 'image_only']:
        accuracy, f1, cm = evaluate(model_to_test, test_loader, device, mode=mode)
        results[mode] = {'accuracy': accuracy, 'f1_score': f1, 'confusion_matrix': cm}
        logging.info(f"--- 模式: {mode} ---")
        logging.info(f"  准确率 (Accuracy): {accuracy:.4f}")
        logging.info(f"  F1 分数 (Macro F1): {f1:.4f}")
        logging.info(f"  混淆矩阵:\n{cm}")
        
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['HAHV', 'HALV', 'LALV', 'LAHV'], 
                        yticklabels=['HAHV', 'HALV', 'LALV', 'LAHV'])
            plt.xlabel('Predicted Label')
            plt.ylabel('Actual Label')
            plt.title(f'Confusion Matrix ({mode}) on Validation Set (BS={FINAL_BATCH_SIZE})') 
            plt.savefig(f'confusion_matrix_{mode}_val_set.png') 
            logging.info(f"混淆矩阵图片已保存为 confusion_matrix_{mode}_val_set.png")
            plt.close() 
        except Exception as e:
            logging.error(f"绘制或保存混淆矩阵失败 ({mode}): {e}")

    logging.info("--- 测试完成 ---")
    print("\n--- 最终结果 (在 20% 验证集上测试) ---")
    print(f"(使用的 Batch Size: {FINAL_BATCH_SIZE})")
    for mode, metrics in results.items():
        print(f"模式: {mode}")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  F1分数: {metrics['f1_score']:.4f}")
        print(f"  (混淆矩阵图片已保存)")

if __name__ == "__main__":
    main()