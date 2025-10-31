import torch
import torch.nn as nn
import torch.optim as optim
from transformers import SwinModel, AutoImageProcessor
import os
import sys
import logging
import gc

# --- DDP修改: 导入 DDP 相关模块 ---
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# --- 1. 导入您已修正的 model.py ---
try:
    from model import MultiModalClassifier
except ImportError:
    print("错误: 找不到 'model.py'。请将此脚本与 'model.py' 放在同一目录。")
    sys.exit(1)

# --- 2. 设置日志 (无修改) ---
def setup_logging(rank=0):
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    # --- DDP修改: 在日志中包含 rank ---
    formatter = logging.Formatter(f'%(asctime)s - RANK_{rank} - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    
    # --- DDP修改: 只在 rank 0 上添加 handler ---
    if rank == 0:
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())

# --- 3. 从您的 train_fast.py 中复制超参数 (无修改) ---
NUM_CLASSES = 4 
NUM_INSTANCES = 10 
NUM_SELECT = 3     
MODEL_PATH = "./local_swin_model/" 
EEG_CHANNELS = 32
EEG_SAMPLES = 384
IMAGE_SIZE = 224

# --- DDP修改: 主逻辑变为 main_worker ---
def find_max_bs_worker(local_rank, world_size, result_queue, mode):
    
    # --- DDP修改: DDP 进程组设置 ---
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358' # 使用不同于训练的端口
    dist.init_process_group("nccl", rank=local_rank, world_size=world_size)
    
    # --- DDP修改: 设备设置 ---
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    setup_logging(local_rank)

    # --- 关键修改：CPU-First 加载逻辑 ---
    
    # 1. Swin 加载到 CPU (所有进程都加载，但只在 rank 0 记录)
    if local_rank == 0: logging.info("正在加载 Swin Transformer (到 CPU)...")
    try:
        swin_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
        swin_model = SwinModel.from_pretrained(MODEL_PATH) 
    except Exception as e:
        if local_rank == 0: logging.error(f"加载 Swin Transformer 失败: {e}")
        return 0

    # 2. 完整模型在 CPU 初始化
    if local_rank == 0: logging.info(f"初始化 MultiModalClassifier (在 CPU 上)...")
    model_on_cpu = MultiModalClassifier(
        swin_processor=swin_processor,
        swin_model=swin_model,
        num_classes=NUM_CLASSES,
        num_select=NUM_SELECT,
        num_instances=NUM_INSTANCES
    ) 

    # 3. 将模型移动到 *指定* GPU
    model = model_on_cpu.to(device)
    
    # 4. 使用 DDP 包装
    # find_unused_parameters 设为 True 确保所有参数被同步
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) 
    
    if local_rank == 0: logging.info("模型已在所有 GPU 上并由 DDP 包装。")

    # 5. 设置优化器和损失 (仅 train 模式需要)
    if mode == 'train':
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss().to(device)
        model.train()
        if local_rank == 0: logging.info("模式: 'train' (将测试 forward + backward + optimizer.step)")
    else:
        model.eval()
        if local_rank == 0: logging.info("模式: 'test' (将测试 'with torch.no_grad():' 中的 forward)")

    # 6. 循环查找最大 Batch Size *Per GPU*
    current_bs_per_gpu = 1
    max_bs_per_gpu = 0
    
    while True:
        total_bs = current_bs_per_gpu * world_size
        if local_rank == 0:
            logging.info(f"--- 正在尝试 BS_Per_GPU = {current_bs_per_gpu} (Total BS = {total_bs}) ---")
        
        try:
            # 1. 创建 *本地* 模拟数据
            eeg_data = torch.rand(current_bs_per_gpu, EEG_CHANNELS, EEG_SAMPLES, dtype=torch.float32).to(device)
            images_data = torch.randint(0, 255, 
                                        (current_bs_per_gpu, NUM_INSTANCES, IMAGE_SIZE, IMAGE_SIZE, 3), 
                                        dtype=torch.uint8).to(device)
            labels = torch.randint(0, NUM_CLASSES, (current_bs_per_gpu,), dtype=torch.long).to(device)

            # 2. 运行模拟
            if mode == 'train':
                optimizer.zero_grad()
                outputs = model(eeg_data, images_data)
                loss = criterion(outputs, labels)
                
                # --- DDP修改: 移除 DP 的 .mean() 检查 ---
                
                loss.backward() # <-- OOM 最可能发生在这里
                optimizer.step()
            else: # mode == 'test'
                with torch.no_grad():
                    outputs = model(eeg_data, images_data) # <-- OOM 可能发生在这里
            
            # 3. 如果成功
            if local_rank == 0: logging.info(f"BS_Per_GPU {current_bs_per_gpu} 成功。")
            max_bs_per_gpu = current_bs_per_gpu
            current_bs_per_gpu += 1
            
            del eeg_data, images_data, labels, outputs
            if mode == 'train':
                del loss
            gc.collect()
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if local_rank == 0: logging.warning(f"BS_Per_GPU {current_bs_per_gpu} 导致 OutOfMemoryError!")
            gc.collect()
            torch.cuda.empty_cache()
            break
        
        except Exception as e:
            if local_rank == 0: logging.error(f"BS_Per_GPU {current_bs_per_gpu} 发生意外错误: {e}")
            break

    # --- DDP修改: 清理并返回结果 ---
    dist.destroy_process_group()
    
    # 只让 rank 0 进程通过 Queue 返回结果
    if local_rank == 0:
        result_queue.put(max_bs_per_gpu)

# --- DDP修改: main 函数作为启动器 ---
def main():
    if len(sys.argv) != 2:
        print("用法: python find_max_batch_size.py [train|test]")
        sys.exit(1)
    
    test_mode = sys.argv[1].lower()
    
    if not torch.cuda.is_available():
        print("未检测到 CUDA! 此脚本需要 GPU。")
        sys.exit(1)

    world_size = torch.cuda.device_count()
    visible_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
    
    # --- DDP修改: 使用 Queue 在进程间通信 ---
    result_queue = mp.Queue()
    
    mp.spawn(find_max_bs_worker,
             args=(world_size, result_queue, test_mode),
             nprocs=world_size,
             join=True)
             
    # 从 Queue 中获取结果
    max_bs_per_gpu = result_queue.get()
    
    print("\n" + "="*40)
    print(f"--- DDP 显存测试完成 ---")
    print(f"模式: '{test_mode}'")
    print(f"可见 GPUs: {visible_gpus} ({world_size} 块)")
    print(f"最大安全 Batch Size (每 GPU): {max_bs_per_gpu}")
    print(f"最大安全 Batch Size (总和): {max_bs_per_gpu * world_size}")
    print("="*40 + "\n")
    print(f"请将 'train_fast.py' 中的 'BATCH_SIZE_PER_GPU' 设置为 {max_bs_per_gpu}")
    if test_mode == 'train':
        print(f"请将 'test.py' 中的 'BATCH_SIZE' 设置为 (运行 'python find_max_batch_size.py test' 找到的值)")

if __name__ == "__main__":
    main()