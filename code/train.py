import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.optim.lr_scheduler as lr_scheduler
from transformers import SwinModel, AutoImageProcessor
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import logging
import sys

# --- 1. 导入已修正的 model.py ---
from model import MultiModalClassifier 

# --- 2. 定义全局 device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 定义超参数 ---
ALL_SUBJECTS = [s for s in range(1, 23) if s != 11] # s01-s10, s12-s22 (共 21 人)

EPOCHS = 100 
LEARNING_RATE = 1e-5
NUM_CLASSES = 4 
NUM_INSTANCES = 10 
NUM_SELECT = 3     
NUM_FOLDS = 10 
RANDOM_SEED = 42 
MODEL_PATH = "./local_swin_model/" 

# --- 关键修改：梯度累积 (Gradient Accumulation) ---
# 解决单卡 OOM 的问题
BATCH_SIZE = 8  # 物理 Batch Size (如果 4 还 OOM，就改成 2)
ACCUMULATION_STEPS = 2 # 累积步数 (有效 Batch Size = 4 * 4 = 16)

# --- 4. 设置日志记录 (Logging) ---
def setup_logging(log_file='train.log'): # 日志文件名已更新
    logging.getLogger().handlers = [] 
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    logging.info("--- 日志系统已启动 ---")

# --- 5. DEAP 数据集类 (无修改) ---
class DEAPDataset(Dataset):
    def __init__(self, subject_list, eeg_dir, face_dir, processor, num_instances=10):
        self.eeg_data = []
        self.labels = []
        self.image_bags = [] 
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

            subject_eeg_data = np.load(eeg_file)
            subject_labels = np.load(label_file)
            
            self.eeg_data.append(subject_eeg_data)
            self.labels.append(subject_labels)

            for trial_idx in range(40): # 40 trials
                trial_str = f'{subject_str}_trial{trial_idx+1:02}'
                frame_files = sorted(glob.glob(os.path.join(face_dir, subject_str, f'{trial_str}_frame_*.png')))
                
                if not frame_files:
                    logging.warning(f"找不到 {trial_str} 的帧，将跳过此 trial。")
                    for _ in range(20): 
                        self.image_bags.append([])
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
                        logging.warning(f"Segment {segment_idx} (T{trial_idx+1}) for {subject_str} 无帧。")
                        segment_frame_files = [frame_files[0]] 
                    
                    self.image_bags.append(segment_frame_files)

        self.eeg_data = np.concatenate(self.eeg_data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        
        logging.info(f"总共加载 {len(self.labels)} 个样本。")
        assert len(self.eeg_data) == len(self.labels) == len(self.image_bags)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg_segment = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        image_bag_files = self.image_bags[idx]
        
        num_frames_in_bag = len(image_bag_files)
        
        if num_frames_in_bag == 0:
            logging.error(f"索引 {idx} 处的图像 bag 为空！将使用空白图像。")
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
                    logging.error(f"加载图像 {img_path} 失败: {e}. 将使用空白图像。")
                    image_bag_np.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        images_tensor = torch.tensor(np.stack(image_bag_np), dtype=torch.uint8) 
        return eeg_segment, images_tensor, label

# --- 6. 训练和验证函数 (修改 train_epoch 以支持梯度累积) ---
def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    train_loop = tqdm(dataloader, desc="Training", leave=False)
    for i, (eeg_data, images_data, labels) in enumerate(train_loop):
        eeg_data = eeg_data.to(device)
        images_data = images_data.to(device) 
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(eeg_data, images_data)
        loss = criterion(outputs, labels)
        
        # 标准化 loss
        loss = loss / accumulation_steps
        
        # 反向传播 (累积梯度)
        loss.backward()
        
        total_loss += loss.item() 
        
        # 只在累积了足够步数后才更新
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()  # 更新模型参数
            optimizer.zero_grad() # 清空梯度
        
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 乘以 accumulation_steps 以报告“真实”的 epoch loss
    avg_loss = (total_loss / len(dataloader)) * accumulation_steps
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    val_loop = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for eeg_data, images_data, labels in val_loop:
            eeg_data = eeg_data.to(device)
            images_data = images_data.to(device)
            labels = labels.to(device)
            
            outputs = model(eeg_data, images_data)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy

# --- 7. 主训练循环 (K-Fold) ---
def main():
    setup_logging('train.log') # 日志名已更新
    
    logging.info(f"设备: {device}")
    
    # Swin 先加载到 CPU
    logging.info("正在从本地加载 Swin Transformer (到 CPU)...")
    swin_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    swin_model = SwinModel.from_pretrained(MODEL_PATH) 
    logging.info("Swin Transformer 加载完毕。")
    
    logging.info("初始化完整数据集 (Subject-Dependent)...")
    
    eeg_dir = './EEGData'
    face_dir = './faces'
    
    # 加载数据 (在 CPU RAM 中)
    full_dataset = DEAPDataset(ALL_SUBJECTS, eeg_dir, face_dir, swin_processor, NUM_INSTANCES)
    
    # 数据加载完毕后，再将 Swin 移到 GPU
    logging.info("正在将 Swin Transformer 移动到 GPU...")
    swin_model = swin_model.to(device)
    logging.info("Swin Transformer 已在 GPU 上。")
    
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_results = [] 

    logging.info(f"--- 开始 {NUM_FOLDS} 折交叉验证 (Subject-Dependent) ---")
    logging.info(f"物理 BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"梯度累积步数: {ACCUMULATION_STEPS}")
    logging.info(f"有效 BATCH_SIZE: {BATCH_SIZE * ACCUMULATION_STEPS}")

    for fold, (train_indices, val_indices) in enumerate(kfold.split(full_dataset)):
        logging.info(f"========== FOLD {fold + 1}/{NUM_FOLDS} ==========")
        
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        logging.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

        logging.info("重新初始化模型、优化器和学习率衰减...")
        
        model = MultiModalClassifier(
            swin_processor=swin_processor,
            swin_model=swin_model, # 传递已在 GPU 上的 Swin
            device=device,
            num_classes=NUM_CLASSES,
            num_select=NUM_SELECT,
            num_instances=NUM_INSTANCES
        ).to(device)
        
        # 移除 DataParallel，我们现在在单卡上运行
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer.zero_grad() # 梯度清零
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
        
        best_val_accuracy = 0.0
        model_save_path = f'./best_model_fold_{fold+1}.pth'

        for epoch in range(EPOCHS):
            logging.info(f"--- Fold {fold+1}, Epoch {epoch+1}/{EPOCHS} ---")
            
            # 传递 ACCUMULATION_STEPS
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, ACCUMULATION_STEPS)
            logging.info(f"Fold {fold+1}, Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            logging.info(f"Fold {fold+1}, Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            scheduler.step()
            logging.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.8f}")
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Fold {fold+1} 新的最佳模型已保存，准确率: {best_val_accuracy:.4f} (路径: {model_save_path})")

        logging.info(f"========== FOLD {fold + 1} 完成 ========== ")
        logging.info(f"此折最佳验证准确率: {best_val_accuracy:.4f}")
        fold_results.append(best_val_accuracy)

    logging.info("--- 所有 K-Fold 训练完毕 ---")
    
    avg_accuracy = np.mean(fold_results)
    std_accuracy = np.std(fold_results)
    
    logging.info(f"所有折的最佳准确率: {fold_results}")
    logging.info(f"平均验证准确率 (10-fold): {avg_accuracy:.4f} +/- {std_accuracy:.4f}")
    logging.info("训练结束。")

if __name__ == "__main__":
    main()