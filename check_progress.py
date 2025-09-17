import psutil
import os

# Check CPU usage
cpu_percent = psutil.cpu_percent(interval=1)
print(f"CPU Usage: {cpu_percent}%")

# Check memory usage
memory = psutil.virtual_memory()
print(f"Memory Usage: {memory.percent}%")

# Check if Python process is running
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
    if 'python' in proc.info['name'].lower():
        print(f"Python Process: PID {proc.info['pid']}, CPU {proc.info['cpu_percent']}%")