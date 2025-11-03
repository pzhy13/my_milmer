import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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
LEARNING_RATE = 1e-5
NUM_CLASSES = 4 
NUM_INSTANCES = 10 
NUM_SELECT = 3     
RANDOM_SEED = 42 
MODEL_PATH = "./local_swin_model/" 

# --- 关键修改：BATCH_SIZE 现在是 *每个 GPU* 的 Batch Size ---
# 假设你 find_max_batch_size 测出 train 模式下每卡可以跑 2
# 3 块 GPU 的总 Batch Size 将是 2 * 3 = 6
BATCH_SIZE_PER_GPU = 128 # <--- 警告：这是你需要用新脚本找到的值
# 梯度累积（如果需要，可以设为 1）
ACCUMULATION_STEPS = 1 # 假设 DDP 提供了足够的总批量

# --- 4. 设置日志记录 (Logging) ---
# --- DDP修改: 日志记录只在 rank 0 进程设置 ---
def setup_logging(rank, log_file='train_ddp.log'):
    logger = logging.getLogger()
    logger.handlers = [] # 清除已有处理器
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(f'%(asctime)s - RANK_{rank} - %(levelname)s - %(message)s')    
    if rank == 0:
        # 只在 Rank 0 上写入文件
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Rank 0 也在控制台输出
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        # 其他 Rank 只设置一个空处理器，避免 "no handler" 警告
        logger.addHandler(logging.NullHandler())

# --- 5. DEAP 数据集类 (无修改) ---
class DEAPDataset(Dataset):
    # ... (此处省略，与原文件相同) ...
    def __init__(self, subject_list, eeg_dir, face_dir, processor, num_instances=10, rank=0):
        self.eeg_data = []
        self.labels = []
        self.image_bags = [] 
        self.processor = processor
        self.num_instances = num_instances
        
        # 只在 rank 0 显示 tqdm 进度条
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
            for trial_idx in range(40): # 40 trials
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
                for segment_idx in range(20): # 20 segments
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


# --- 6. 训练和验证函数 ---
# --- DDP修改: 移除 DataParallel 的 loss.mean() 逻辑 ---
# --- DDP修改: 返回局部的 preds 和 labels, 在 main_worker 中聚合 ---
def train_epoch(model, dataloader, optimizer, criterion, device, accumulation_steps, rank):
    model.train()
    total_loss = 0
    local_preds = []
    local_labels = []
    
    # --- DDP修改: Dataloader 包装在 tqdm 中 (只在 rank 0 显示) ---
    if rank == 0:
        train_loop = tqdm(dataloader, desc="Training", leave=False)
    else:
        train_loop = dataloader

    for i, (eeg_data, images_data, labels) in enumerate(train_loop):
        # 数据移动到 *当前进程* 分配的 GPU
        eeg_data = eeg_data.to(device) 
        images_data = images_data.to(device) 
        labels = labels.to(device)
        
        outputs = model(eeg_data, images_data)
        loss = criterion(outputs, labels)
        
        # DDP 不需要 .mean()，loss 已经是标量
            
        loss = loss / accumulation_steps
        loss.backward() # DDP 会在此处自动同步梯度
        
        total_loss += loss.item() 
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        preds = torch.argmax(outputs, dim=1)
        local_preds.append(preds.cpu())
        local_labels.append(labels.cpu())
        
    avg_loss = (total_loss / len(dataloader)) * accumulation_steps
    
    # 返回 *局部的* loss, preds, labels
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
            
            # DDP 不需要 .mean()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            local_preds.append(preds.cpu())
            local_labels.append(labels.cpu())
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss, torch.cat(local_preds), torch.cat(local_labels)

# --- DDP修改: 用于在所有 GPU 间聚合指标的辅助函数 ---
def aggregate_metrics(local_loss, local_preds, local_labels, world_size):
    # 1. 聚合 Loss (取平均)
    loss_tensor = torch.tensor(local_loss).cuda()
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    global_loss = loss_tensor.item() / world_size

    # 2. 聚合 Preds 和 Labels (收集所有)
    # 创建收集列表
    preds_list = [torch.zeros_like(local_preds) for _ in range(world_size)]
    labels_list = [torch.zeros_like(local_labels) for _ in range(world_size)]
    
    # 从所有 rank 收集
    dist.all_gather(preds_list, local_preds.cuda())
    dist.all_gather(labels_list, local_labels.cuda())

    # 仅在 rank 0 上计算全局准确率
    global_accuracy = 0.0
    if dist.get_rank() == 0:
        all_preds = torch.cat(preds_list).cpu()
        all_labels = torch.cat(labels_list).cpu()
        global_accuracy = accuracy_score(all_labels.numpy(), all_preds.numpy())
        
    return global_loss, global_accuracy

# --- 7. 主函数 (重构为 DDP 的 main_worker) ---
def main_worker(local_rank, world_size):
    # --- DDP修改: 设置进程组 ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # 确保端口未被占用
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    
    # --- DDP修改: 设置日志 ---
    setup_logging(local_rank, 'train_ddp.log')
    
    # --- DDP修改: 将此进程绑定到特定 GPU ---
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        logging.info(f"--- DDP 训练启动, World Size (GPU数量): {world_size} ---")

    # --- 关键修改：Swin 严格加载到 CPU ---
    if local_rank == 0: logging.info("正在从本地加载 Swin Transformer (到 CPU)...")
    swin_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    swin_model = SwinModel.from_pretrained(MODEL_PATH) # 不加 .to(device)
    if local_rank == 0: logging.info("Swin Transformer 加载完毕 (在 CPU)。")
    
    if local_rank == 0: logging.info("初始化完整数据集...")
    eeg_dir = './EEGData'
    face_dir = './faces'
    # --- DDP修改: 传入 rank 以控制日志/tqdm ---
    full_dataset = DEAPDataset(ALL_SUBJECTS, eeg_dir, face_dir, swin_processor, NUM_INSTANCES, rank=local_rank)
    
    if local_rank == 0: logging.info("正在创建 80/20 训练/验证集...")
    total_size = len(full_dataset)
    val_size = int(total_size * 0.2) 
    train_size = total_size - val_size 
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    if local_rank == 0: logging.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # --- DDP修改: 使用 DistributedSampler ---
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    
    # --- DDP修改: shuffle=False (sampler 负责)
    # --- DDP修改: BATCH_SIZE 是 BATCH_SIZE_PER_GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_PER_GPU, shuffle=False, num_workers=4, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_PER_GPU, shuffle=False, num_workers=4, pin_memory=True, sampler=val_sampler)
    
    # --- 关键修改：模型初始化严格在 CPU ---
    if local_rank == 0: logging.info("初始化模型架构 (在 CPU)...")
    model_on_cpu = MultiModalClassifier(
        swin_processor=swin_processor,
        swin_model=swin_model,
        num_classes=NUM_CLASSES,
        num_select=NUM_SELECT,
        num_instances=NUM_INSTANCES
    ) 

    # --- DDP修改: 将模型移动到 *指定* GPU ---
    model = model_on_cpu.to(device)
    
    # --- DDP修改: 使用 DDP 包装模型 ---
    # find_unused_parameters 设为 True 确保所有参数被同步 (如果模型有分支)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) 
    
    if local_rank == 0:
        total_gpus = world_size
        logging.info(f"--- 开始单次训练 (Subject-Dependent) ---")
        logging.info(f"物理 BATCH_SIZE (每 GPU): {BATCH_SIZE_PER_GPU}")
        logging.info(f"梯度累积步数: {ACCUMULATION_STEPS}")
        logging.info(f"总 GPU 数量: {total_gpus}")
        logging.info(f"有效 BATCH_SIZE (全局): {BATCH_SIZE_PER_GPU * total_gpus * ACCUMULATION_STEPS}")

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer.zero_grad() 
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
    
    best_val_accuracy = 0.0
    model_save_path = './best_model_fast_run_ddp.pth' # 新的保存路径

    for epoch in range(EPOCHS):
        # --- DDP修改: 必须设置 epoch 以确保 sampler 正确 shuffle ---
        train_sampler.set_epoch(epoch)
        
        if local_rank == 0: logging.info(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 1. 训练
        local_train_loss, local_train_preds, local_train_labels = train_epoch(model, train_loader, optimizer, criterion, device, ACCUMULATION_STEPS, local_rank)
        # --- DDP修改: 聚合训练指标 ---
        train_loss, train_acc = aggregate_metrics(local_train_loss, local_train_preds, local_train_labels, world_size)
        
        if local_rank == 0:
            logging.info(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # 2. 验证
        local_val_loss, local_val_preds, local_val_labels = validate_epoch(model, val_loader, criterion, device, local_rank)
        # --- DDP修改: 聚合验证指标 ---
        val_loss, val_acc = aggregate_metrics(local_val_loss, local_val_preds, local_val_labels, world_size)
        
        if local_rank == 0:
            logging.info(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        # --- DDP修改: 只在 rank 0 进程保存 ---
        if local_rank == 0:
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                
                # 保存时解开 DDP 包装 (与 DP 相同, 都是 .module)
                torch.save(model.module.state_dict(), model_save_path)
                logging.info(f"新的最佳模型已保存，准确率: {best_val_accuracy:.4f} (路径: {model_save_path})")

    if local_rank == 0:
        logging.info("--- 训练完毕 ---")
        logging.info(f"最佳验证准确率: {best_val_accuracy:.4f}")
        
    # --- DDP修改: 清理进程组 ---
    dist.destroy_process_group()

# --- DDP修改: main 函数作为启动器 ---
def main():
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("未检测到 CUDA 设备，DDP 训练需要 GPU。")
        return
        
    # 使用 mp.spawn 启动 DDP
    # 它会为每个 GPU (nprocs=world_size) 调用 main_worker
    # 并自动传入 local_rank (0, 1, 2...) 和 world_size
    mp.spawn(main_worker,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()