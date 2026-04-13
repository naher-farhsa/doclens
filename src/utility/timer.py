import time
import sys

class PipelineTimer:
    """A context manager to time specific pipeline steps and print to console immediately."""
    
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        print(f"\n⏱️  [START] {self.step_name}...", flush=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            print(f"⏱️  [DONE] {self.step_name} completed in {elapsed:.2f} seconds.", flush=True)
        else:
            print(f"⏱️  [ERROR] {self.step_name} failed after {elapsed:.2f} seconds.", flush=True)
