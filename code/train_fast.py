import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torch.optim.lr_scheduler as lr_scheduler
from transformers import SwinModel, AutoImageProcessor
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import logging
import sys
# --- 新增: 导入 torchvision 用于数据增强 ---
from torchvision import transforms 

# --- DDP修改: 导入 DDP 相关模块 ---
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# --- 1. 导入已修正的 model.py ---
from model import MultiModalClassifier 

# --- 2. 定义全局 device (将在 main_worker 中被覆盖) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 3. 定义超参数 ---
ALL_SUBJECTS = [s for s in range(1, 23) if s != 11] 
EPOCHS = 100 
LEARNING_RATE = 2e-5 
WARMUP_EPOCHS = 5    
NUM_CLASSES = 4 
NUM_INSTANCES = 10 
NUM_SELECT = 3     
RANDOM_SEED = 42 
MODEL_PATH = "./local_swin_model/" # 请确保此路径正确

# --- 新增 EEG 参数配置 ---
EEG_CHANNELS = 32 
EEG_TIME_PTS = 384  
EEG_SIZE_TOTAL = EEG_CHANNELS * EEG_TIME_PTS 

BATCH_SIZE_PER_GPU = 128 
ACCUMULATION_STEPS = 1 

# --- 4. 设置日志记录 (Logging) ---
def setup_logging(rank, log_file='train_ddp_final.log'): 
    logger = logging.getLogger()
    logger.handlers = [] 
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s - RANK_{rank} - %(levelname)s - %(message)s')      
    if rank == 0:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())

# --- 5. DEAP 数据集类 (无修改) ---
class DEAPDataset(Dataset):
    def __init__(self, subject_list, eeg_dir, face_dir, processor, num_instances=10, rank=0):
        self.eeg_data = []
        self.labels = []
        self.image_bags = [] 
        self.processor = processor
        self.num_instances = num_instances
        
        if rank == 0:
            logging.info(f"开始为 {len(subject_list)} 个被试加载数据...")
            subject_loop = tqdm(subject_list)
        else:
            subject_loop = subject_list

        for subject_id in subject_loop:
            subject_str = f's{subject_id:02}'
            eeg_file = os.path.join(eeg_dir, f'{subject_str}_eeg.npy')
            label_file = os.path.join(eeg_dir, f'{subject_str}_labels.npy')
            if not (os.path.exists(eeg_file) and os.path.exists(label_file)):
                if rank == 0: logging.warning(f"找不到被试 {subject_str} 的数据，已跳过。")
                continue
            subject_eeg_data = np.load(eeg_file)
            subject_labels = np.load(label_file)
            self.eeg_data.append(subject_eeg_data)
            self.labels.append(subject_labels)
            for trial_idx in range(40): 
                trial_str = f'{subject_str}_trial{trial_idx+1:02}'
                frame_files = sorted(glob.glob(os.path.join(face_dir, subject_str, f'{trial_str}_frame_*.png')))
                if not frame_files:
                    for _ in range(20): 
                        self.image_bags.append([])
                    continue
                total_frames = len(frame_files)
                frames_per_segment_actual = total_frames // 20 
                if frames_per_segment_actual == 0: 
                    frames_per_segment_actual = total_frames
                for segment_idx in range(20): 
                    start_frame_idx = segment_idx * frames_per_segment_actual
                    end_frame_idx = (segment_idx + 1) * frames_per_segment_actual
                    if segment_idx == 19: 
                        end_frame_idx = total_frames
                    segment_frame_files = frame_files[start_frame_idx:end_frame_idx]
                    if not segment_frame_files:
                        segment_frame_files = [frame_files[0]] if frame_files else [] 
                    self.image_bags.append(segment_frame_files)
        self.eeg_data = np.concatenate(self.eeg_data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        if rank == 0: logging.info(f"总共加载 {len(self.labels)} 个样本。")
        assert len(self.eeg_data) == len(self.labels) == len(self.image_bags)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        eeg_segment = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        image_bag_files = self.image_bags[idx]
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


# === 关键修改: 将 TransformWrapper 移到顶层 ===
# 修复: AttributeError: Can't pickle local object
class TransformWrapper(Dataset):
    """
    一个包装器，用于在数据集被 random_split 后，对训练子集应用图像增强。
    """
    def __init__(self, subset: Subset, transform=None):
        self.subset = subset
        self.transform = transform
        # 确保原始数据集的 num_instances 被保留
        self.num_instances = subset.dataset.num_instances 

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # subset[idx] 返回的是 (eeg_segment, images_tensor, label)
        eeg, imgs, label = self.subset[idx]
        
        # imgs: (NUM_INSTANCES, 224, 224, 3) uint8 tensor
        if self.transform:
            new_imgs = []
            # 遍历 bag 中的每个实例
            for i in range(imgs.shape[0]):
                # 将 (H, W, C) uint8 Tensor 转换为 PIL Image (需要 C, H, W)
                pil_img = transforms.ToPILImage()(imgs[i].permute(2, 0, 1))
                
                # 应用增强
                aug_img = self.transform(pil_img)
                
                # 将增强后的 PIL Image 转换回 (H, W, C) numpy 数组
                # 注意: np.array(PIL Image) 默认返回 (H, W, C)
                new_imgs.append(np.array(aug_img))
            
            # 堆叠回 Tensor
            imgs = torch.tensor(np.stack(new_imgs), dtype=torch.uint8)
            
        return eeg, imgs, label
# ===============================================


# --- 6. 训练和验证函数 (无修改) ---
def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps, rank):
    model.train()
    total_loss = 0
    local_preds = []
    local_labels = []
    
    if rank == 0:
        train_loop = tqdm(dataloader, desc="Training", leave=False)
    else:
        train_loop = dataloader

    for i, (eeg_data, images_data, labels) in enumerate(train_loop):
        eeg_data = eeg_data.to(device) 
        images_data = images_data.to(device) 
        labels = labels.to(device)
        
        outputs = model(eeg_data, images_data)
        loss = criterion(outputs, labels)
        
        loss = loss / accumulation_steps
        loss.backward() 
        
        total_loss += loss.item() 
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        preds = torch.argmax(outputs, dim=1)
        local_preds.append(preds.cpu())
        local_labels.append(labels.cpu())
        
    avg_loss = (total_loss / len(dataloader)) * accumulation_steps
    return avg_loss, torch.cat(local_preds), torch.cat(local_labels)

def validate_epoch(model, dataloader, criterion, device, rank):
    model.eval()
    total_loss = 0
    local_preds = []
    local_labels = []

    if rank == 0:
        val_loop = tqdm(dataloader, desc="Validating", leave=False)
    else:
        val_loop = dataloader
        
    with torch.no_grad():
        for eeg_data, images_data, labels in val_loop:
            eeg_data = eeg_data.to(device)
            images_data = images_data.to(device)
            labels = labels.to(device)
            
            outputs = model(eeg_data, images_data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            local_preds.append(preds.cpu())
            local_labels.append(labels.cpu())
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss, torch.cat(local_preds), torch.cat(local_labels)

def aggregate_metrics(local_loss, local_preds, local_labels, world_size):
    loss_tensor = torch.tensor(local_loss).cuda()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_loss = loss_tensor.item() / world_size

    local_preds_cuda = local_preds.cuda()
    local_labels_cuda = local_labels.cuda()

    preds_list = [torch.zeros_like(local_preds_cuda) for _ in range(world_size)] 
    labels_list = [torch.zeros_like(local_labels_cuda) for _ in range(world_size)]
    
    dist.all_gather(preds_list, local_preds_cuda)
    dist.all_gather(labels_list, local_labels_cuda)

    global_accuracy = 0.0
    if dist.get_rank() == 0:
        all_preds = torch.cat(preds_list).cpu()
        all_labels = torch.cat(labels_list).cpu()
        global_accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
        
    return global_loss, global_accuracy

# --- 7. 主函数 (重构为 DDP 的 main_worker) ---
def main_worker(local_rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' 
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    
    setup_logging(local_rank, 'train_ddp_final.log')
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        logging.info(f"--- DDP 训练启动, World Size (GPU数量): {world_size} ---")

    swin_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    swin_model = SwinModel.from_pretrained(MODEL_PATH) 
    
    # --- 定义数据增强 ---
    train_transforms = transforms.Compose([
        # 强制将 PIL Image 转换为 Tensor 再处理会更复杂，这里只用 PIL Image 友好的 Transforms
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])
    
    if local_rank == 0: logging.info("初始化完整数据集...")
    eeg_dir = './EEGData' # 请确保此路径正确
    face_dir = './faces'  # 请确保此路径正确
    
    # 初始化完整的 DEAPDataset
    full_dataset = DEAPDataset(ALL_SUBJECTS, eeg_dir, face_dir, swin_processor, NUM_INSTANCES, rank=local_rank)
    
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2) 
    train_size = total_size - val_size 
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    
    # random_split 返回的是 Subset
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # --- 使用顶层定义的 TransformWrapper 来包裹训练集子集，应用增强 ---
    train_dataset = TransformWrapper(train_subset, transform=train_transforms)
    
    # --- 验证集子集保持不变 (不应用增强) ---
    val_dataset = TransformWrapper(val_subset, transform=None)
    
    if local_rank == 0: logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # --- DDP修改: 使用 DistributedSampler ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    
    # <--- 关键修改: 增加 num_workers 解决 IO 瓶颈
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU, shuffle=False, num_workers=2, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, shuffle=False, num_workers=2, pin_memory=True, sampler=val_sampler)
    
    if local_rank == 0: logging.info("初始化模型架构 (在 CPU)...")
    
    # --- 传入 NervFormer 相关参数 ---
    model_on_cpu = MultiModalClassifier(
        swin_processor=swin_processor,
        swin_model=swin_model,
        num_classes=NUM_CLASSES,
        num_select=NUM_SELECT,
        num_instances=NUM_INSTANCES,
        use_nerv_eeg=True,        
        eeg_channels=EEG_CHANNELS,
        eeg_time_len=EEG_TIME_PTS,
        # ==================================================
        # [重要修改] 提高 Dropout 率，配合 Swin 冻结策略
        # 从 0.3 提高到 0.5，抑制 NervFormer 过拟合
        # ==================================================
        transformer_dropout_rate=0.5,
        cls_dropout_rate=0.5
    ) 

    model = model_on_cpu.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) 
    
    if local_rank == 0:
        total_gpus = world_size
        logging.info(f"--- 开始单次训练 (Subject-Dependent) ---")
        logging.info(f"物理 BATCH_SIZE (每 GPU): {BATCH_SIZE_PER_GPU}")
        logging.info(f"有效 BATCH_SIZE (全局): {BATCH_SIZE_PER_GPU * total_gpus * ACCUMULATION_STEPS}")
        logging.info(f"目标学习率: {LEARNING_RATE}")
        logging.info(f"热身 Epochs: {WARMUP_EPOCHS}")
        logging.info(f"正则化: Label Smoothing=0.1, Dropout=0.5 (Increased)")

    # --- 关键修改: 使用 Label Smoothing 防止过拟合 ---
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    optimizer.zero_grad() 
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=0)
    
    best_val_accuracy = 0.0
    model_save_path = './best_model_fast_run_ddp.pth' 

    for epoch in range(EPOCHS):
        train_sampler.set_epoch(epoch)
        
        if epoch < WARMUP_EPOCHS:
            current_lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        if local_rank == 0: 
            logging.info(f"--- Epoch {epoch+1}/{EPOCHS} ---")
            logging.info(f"当前学习率: {optimizer.param_groups[0]['lr']:.8f}")
        
        # 1. 训练
        local_train_loss, local_train_preds, local_train_labels = train_epoch(model, train_loader, optimizer, criterion, device, ACCUMULATION_STEPS, local_rank)
        dist.barrier()
        train_loss, train_acc = aggregate_metrics(local_train_loss, local_train_preds, local_train_labels, world_size)
        
        if local_rank == 0:
            logging.info(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 2. 验证
        local_val_loss, local_val_preds, local_val_labels = validate_epoch(model, val_loader, criterion, device, local_rank)
        dist.barrier()
        val_loss, val_acc = aggregate_metrics(local_val_loss, local_val_preds, local_val_labels, world_size)
        
        if local_rank == 0:
            logging.info(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if epoch >= WARMUP_EPOCHS:
            scheduler.step()
        
        if local_rank == 0:
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.module.state_dict(), model_save_path)
                logging.info(f"新的最佳模型已保存，准确率: {best_val_accuracy:.4f} (路径: {model_save_path})")

    if local_rank == 0:
        logging.info("--- 训练完毕 ---")
        logging.info(f"最佳验证准确率: {best_val_accuracy:.4f}")
        
    dist.destroy_process_group()

# --- DDP修改: main 函数作为启动器 ---
def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("未检测到 CUDA 设备。")
        return
        
    mp.spawn(main_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()