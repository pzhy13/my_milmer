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

from model import MultiModalClassifier 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

ALL_SUBJECTS = [s for s in range(1, 23) if s != 11] 
NUM_CLASSES = 4 
NUM_INSTANCES = 10 
NUM_SELECT = 3     
MODEL_PATH = "./local_swin_model/" 
CHECKPOINT_PATH = './best_model_fast_run_ddp.pth' 
BATCH_SIZE = 12 
EEG_CHANNELS = 32
EEG_TIME_PTS = 384

def setup_logging(log_file='test_multi_gpu.log'): 
    logging.getLogger().handlers = [] 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s  - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logging.info("--- 测试脚本日志系统已启动 ---")

# --- 5. DEAP 数据集类 (完全同步 train_fast.py 的加载逻辑) ---
class DEAPDataset(Dataset):
    def __init__(self, subject_list, eeg_dir, face_dir, processor, num_instances=10):
        self.eeg_data = []
        self.labels = []
        self.image_bags = [] 
        self.sample_info = [] # (subject_id, trial_idx, segment_idx)
        self.processor = processor
        self.num_instances = num_instances
        
        logging.info(f"开始为 {len(subject_list)} 个被试加载数据...")
        
        for subject_id in tqdm(subject_list):
            subject_str = f's{subject_id:02}'
            eeg_file = os.path.join(eeg_dir, f'{subject_str}_eeg.npy')
            label_file = os.path.join(eeg_dir, f'{subject_str}_labels.npy')
            
            if not (os.path.exists(eeg_file) and os.path.exists(label_file)):
                logging.warning(f"找不到被试 {subject_str} 的数据，已跳过。")
                continue
                
            subject_eeg_data = np.load(eeg_file) # (Samples, 32, 384)
            subject_labels = np.load(label_file)   # (Samples,)
            
            # 存储 EEG 和 标签
            self.eeg_data.append(subject_eeg_data)
            self.labels.append(subject_labels)
            
            # --- 修正: 动态生成 sample_info ---
            # train_fast.py 逻辑：先 append eeg, 然后循环 40 trials * 20 segments
            # 因此这里的 subject_eeg_data[i] 对应于下面的 trial/segment 循环顺序
            num_samples = subject_eeg_data.shape[0]
            assert num_samples == 800, f"Subject {subject_str} sample count mismatch: {num_samples}"
            
            current_subject_sample_count = 0
            
            for trial_idx in range(40): # 40 trials
                trial_str = f'{subject_str}_trial{trial_idx+1:02}'
                frame_files = sorted(glob.glob(os.path.join(face_dir, subject_str, f'{trial_str}_frame_*.png')))
                
                if not frame_files:
                    for segment_idx in range(20): 
                        self.image_bags.append([])
                        self.sample_info.append((subject_id, trial_idx, segment_idx))
                        current_subject_sample_count += 1
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
                        segment_frame_files = [frame_files[0]] if frame_files else [] 
                    
                    self.image_bags.append(segment_frame_files)
                    self.sample_info.append((subject_id, trial_idx, segment_idx))
                    current_subject_sample_count += 1
            
            # 确保数量对齐
            if current_subject_sample_count != num_samples:
                logging.error(f"严重错误: 被试 {subject_str} EEG样本数 ({num_samples}) 与 图像Bag数 ({current_subject_sample_count}) 不匹配！")
                
        self.eeg_data = np.concatenate(self.eeg_data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        logging.info(f"总共加载 {len(self.labels)} 个样本。")
        assert len(self.eeg_data) == len(self.labels) == len(self.image_bags) == len(self.sample_info)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_segment = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        image_bag_files = self.image_bags[idx]
        
        # 获取样本元数据
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
                    image_bag_np.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        images_tensor = torch.tensor(np.stack(image_bag_np), dtype=torch.uint8) 
        return eeg_segment, images_tensor, label

def evaluate(model, dataloader, device, mode='multimodal'):
    model.eval() 
    all_preds = []
    all_labels = []

    logging.info(f"评估将在设备 {device} 上运行。")
    eval_loop = tqdm(dataloader, desc=f"Testing ({mode})", leave=False)
    
    with torch.no_grad(): 
        for eeg_data, images_data, labels in eval_loop:
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

def main():
    setup_logging(f'test_final.log') 
    
    logging.info(f"将使用设备: {device}")

    # --- 1. 加载模型 (CPU) ---
    logging.info("正在初始化模型 (CPU)...")
    swin_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    swin_model = SwinModel.from_pretrained(MODEL_PATH)
    
    model_on_cpu = MultiModalClassifier(
        swin_processor=swin_processor,
        swin_model=swin_model, 
        num_classes=NUM_CLASSES,
        num_select=NUM_SELECT,
        num_instances=NUM_INSTANCES,
        use_nerv_eeg=True,
        eeg_channels=EEG_CHANNELS,
        eeg_time_len=EEG_TIME_PTS,
        # 这里的 Dropout 无所谓，因为是 eval 模式，但保持一致更好
        transformer_dropout_rate=0.3,
        cls_dropout_rate=0.3
    )
         
    # --- 2. 加载权重 ---
    logging.info(f"正在从 {CHECKPOINT_PATH} 加载权重...")
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu') 
        new_state_dict = OrderedDict()
        # 检查并移除 DDP 的 'module.' 前缀
        is_data_parallel = False
        for k in checkpoint.keys():
            if k.startswith('module.'):
                is_data_parallel = True
                break
        
        if is_data_parallel:
            logging.info("检测到 'module.' 前缀，正在移除以匹配单机模型...")
            for k, v in checkpoint.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict_to_load = new_state_dict 
        else:
            state_dict_to_load = checkpoint
        
        model_on_cpu.load_state_dict(state_dict_to_load)
        logging.info("模型权重加载成功。")
    except Exception as e:
         logging.error(f"加载模型权重失败: {e}")
         return
         
    # --- 3. 准备数据 ---
    logging.info("初始化完整数据集...")
    eeg_dir = './EEGData'
    face_dir = './faces'
    full_dataset = DEAPDataset(ALL_SUBJECTS, eeg_dir, face_dir, swin_processor, NUM_INSTANCES)
    
    RANDOM_SEED = 42 
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2) 
    train_size = total_size - val_size 
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    logging.info(f"Validation set size: {len(val_dataset)}")
    test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True if device.type=='cuda' else False)

    # --- 4. 移动模型到 GPU 并并行化 ---
    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f"应用 nn.DataParallel ({torch.cuda.device_count()} GPUs)...")
        model_to_test = nn.DataParallel(model_on_cpu)
    else:
        model_to_test = model_on_cpu

    model_to_test.to(device)
    model_to_test.eval()

    # --- 5. 执行评估 ---
    results = {}
    for mode in ['multimodal', 'eeg_only', 'image_only']:
        accuracy, f1, cm = evaluate(model_to_test, test_loader, device, mode=mode)
        results[mode] = {'accuracy': accuracy, 'f1_score': f1, 'confusion_matrix': cm}
        
        logging.info(f"--- 模式: {mode} ---")
        logging.info(f"  准确率: {accuracy:.4f}")
        logging.info(f"  F1 分数: {f1:.4f}")
        
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['HAHV', 'LAHV', 'LALV', 'HALV'], # 0,1,2,3
                        yticklabels=['HAHV', 'LAHV', 'LALV', 'HALV'])
            plt.xlabel('Predicted Label')
            plt.ylabel('Actual Label')
            plt.title(f'Confusion Matrix ({mode})') 
            plt.savefig(f'confusion_matrix_{mode}.png') 
            plt.close() 
        except Exception as e:
            logging.error(f"绘图失败: {e}")

    print("\n--- 最终结果 ---")
    for mode, metrics in results.items():
        print(f"模式: {mode} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()