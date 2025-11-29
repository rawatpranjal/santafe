#!/usr/bin/env python3
"""
Simple monitoring script for curriculum training progress.
"""
import time
import os
from pathlib import Path

log_file = Path("training_log_curriculum.txt")

print("Monitoring curriculum training...")
print(f"Log file: {log_file}")
print("-" * 60)

last_size = 0
no_change_count = 0

while True:
    try:
        if log_file.exists():
            current_size = log_file.stat().st_size
            
            if current_size != last_size:
                # New content, print last 20 lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print("\n" + "=" * 60)
                    print(f"[{time.strftime('%H:%M:%S')}] Log updated ({current_size} bytes)")
                    print("=" * 60)
                    for line in lines[-20:]:
                        print(line, end='')
                    print("\n" + "=" * 60)
                
                last_size = current_size
                no_change_count = 0
            else:
                no_change_count += 1
                if no_change_count % 12 == 0:  # Every minute
                    print(f"[{time.strftime('%H:%M:%S')}] No changes for {no_change_count * 5} seconds...")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Waiting for log file...")
        
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
