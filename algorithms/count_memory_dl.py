import os
import threading
import time

import numpy as np
import psutil
import torch


def get_total_rss(proc):
    """统计主进程 + 所有子进程的 RSS"""
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        total += child.memory_info().rss
    return total

def monitor_memory(stop_event, interval, cpu_mem, gpu_mem):
    proc = psutil.Process(os.getpid())

    while not stop_event.is_set():
        # CPU memory
        cpu_mem.append(get_total_rss(proc))

        # GPU memory（如果使用CUDA）
        if torch.cuda.is_available():
            gpu_mem.append(torch.cuda.memory_allocated())

        time.sleep(interval)
def measure_memory(func, *args, interval=0.01, **kwargs):
    cpu_mem = []
    gpu_mem = []
    stop_event = threading.Event()

    sampler_thread = threading.Thread(
        target=monitor_memory,
        args=(stop_event, interval, cpu_mem, gpu_mem)
    )
    sampler_thread.start()

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    stop_event.set()
    sampler_thread.join()

    cpu_mem = np.array(cpu_mem) / (1024**2)  # MB
    print("========== CPU 内存 ==========")
    print(f"平均: {cpu_mem.mean():.2f} MB")
    # print(f"峰值: {cpu_mem.max():.2f} MB")
    # print(f"标准差: {cpu_mem.std():.2f} MB")

    if gpu_mem:
        gpu_mem = np.array(gpu_mem) / (1024**2)
        print("========== GPU 显存 ==========")
        print(f"平均: {gpu_mem.mean():.2f} MB")
        # print(f"峰值: {gpu_mem.max():.2f} MB")

    # print(f"平均内存占用: {avg_mem:.2f} MB")
    print(f"运行时间: {end_time - start_time:.4f} 秒")

    return cpu_mem.mean(), end_time - start_time