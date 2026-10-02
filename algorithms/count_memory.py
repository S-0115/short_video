import psutil
import os
import threading
import time

def memory_sampler(interval, stop_event, mem_list):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        mem = process.memory_info().rss  # 单位：字节
        mem_list.append(mem)
        time.sleep(interval)

def measure_memory(func, *args, interval=0.01, **kwargs):
    mem_list = []
    stop_event = threading.Event()

    sampler_thread = threading.Thread(
        target=memory_sampler,
        args=(interval, stop_event, mem_list)
    )
    sampler_thread.start()

    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    stop_event.set()
    sampler_thread.join()

    avg_mem = sum(mem_list) / len(mem_list) / (1024 ** 2)  # 转为 MB

    print(f"平均内存占用: {avg_mem:.2f} MB")
    print(f"运行时间: {end_time - start_time:.4f} 秒")

    return avg_mem, end_time - start_time