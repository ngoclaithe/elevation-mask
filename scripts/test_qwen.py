import time
import cv2
import numpy as np
import torch
from PIL import Image

print("Testing Qwen-VL inside container...")
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

from app.settings import settings
print(f"Configured vl_model: {settings.vl_model}")

t0 = time.time()
try:
    from app.pipeline.qwen_vl import critique_overlay
    bgr = cv2.imread("samples/image_1.jpg")
    overlay = cv2.imread("samples/image_1.jpg")  # test dummy overlay
    print("Calling critique_overlay...")
    issues = critique_overlay(bgr, overlay)
    print(f"critique_overlay completed in {time.time() - t0:.2f}s")
    print(f"Found {len(issues)} issues:")
    for iss in issues:
        print(" ", iss.to_dict())
except Exception as e:
    import traceback
    print("Exception during Qwen test:", e)
    traceback.print_exc()
