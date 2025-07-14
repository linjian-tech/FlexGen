import psutil
import os

print(psutil.virtual_memory())
print(f"Current memory usage: {psutil.Process(os.getpid()).memory_info().rss / 124 ** 3:.2f} GB")