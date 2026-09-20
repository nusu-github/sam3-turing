import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import platform
import statistics
import threading
import time
import torch


def configure():
    torch.set_num_threads(4)
    torch.manual_seed(4302)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    torch.cuda.init()


def environment():
    p = torch.cuda.get_device_properties(0)
    return dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=p.name, capability=list(torch.cuda.get_device_capability()),
                physical_vram_bytes=p.total_memory, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                tf32_cudnn=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


class DeviceMemorySampler:
    """Sample total device memory through NVML; a sampled lower bound on peak."""
    def __init__(self):
        import pynvml
        self.nv = pynvml
        self.nv.nvmlInit()
        self.handle = self.nv.nvmlDeviceGetHandleByIndex(0)
        self.stop_event = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.loop, daemon=True)
    def loop(self):
        while not self.stop_event.is_set():
            self.samples.append(self.nv.nvmlDeviceGetMemoryInfo(self.handle).used)
            self.stop_event.wait(.005)
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop_event.set()
        self.thread.join(timeout=2)
    def result(self):
        return dict(sampled_device_used_peak_bytes=max(self.samples,default=0),
                    sampled_device_used_min_bytes=min(self.samples,default=0),
                    samples=len(self.samples), requested_interval_ms=5,
                    scope='Whole GPU NVML sampled usage, not an exact instantaneous peak')


def measure(invoke, warmups=5, repetitions=20):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    with DeviceMemorySampler() as sampler:
        begin=time.perf_counter()
        output=invoke()
        torch.cuda.synchronize()
        cold=time.perf_counter()-begin
        del output
        for _ in range(warmups):
            output=invoke()
            del output
        torch.cuda.synchronize()
        times, gpu_times = [], []
        for index in range(repetitions):
            first,last=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start=time.perf_counter()
            first.record()
            output=invoke()
            last.record()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            gpu_times.append(first.elapsed_time(last)/1000)
            if index < repetitions-1:
                del output
        allocated=torch.cuda.max_memory_allocated()
        reserved=torch.cuda.max_memory_reserved()
        free,total=torch.cuda.mem_get_info()
    result=dict(cold_seconds=cold,warmups=warmups,repetitions=repetitions,
                wall_seconds=times,gpu_seconds=gpu_times,median_wall_seconds=statistics.median(times),
                median_gpu_seconds=statistics.median(gpu_times),
                peak_allocated_bytes=allocated,peak_reserved_bytes=reserved,
                allocated_before_bytes=allocated_before,reserved_before_bytes=reserved_before,
                end_device_used_bytes=total-free,physical_vram_bytes=total,nvml=sampler.result())
    return output,result


def save_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')
