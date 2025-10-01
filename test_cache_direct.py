# 简单测试脚本 test_cache_direct.py
import sys

sys.path.append('/root/GeoCLIP')

from geoclip.datasets.adaclip_adapter import AdaCLIPToGeoCLIPAdapter
import os

print("=== 直接测试缓存功能 ===")

# 创建适配器
adapter = AdaCLIPToGeoCLIPAdapter(
    cache_depth=True,
    depth_cache_dir="./test_cache_direct"
)

# 找一个真实的图像文件
test_image = "/root/GeoCLIP/geoclip/datasets/mvtec_anomaly_detection/bottle/train/good/000.png"

if not os.path.exists(test_image):
    # 如果000.png不存在，找第一个可用的
    import glob

    pattern = "/root/GeoCLIP/geoclip/datasets/mvtec_anomaly_detection/bottle/train/good/*.png"
    images = glob.glob(pattern)
    if images:
        test_image = images[0]
        print(f"使用图像: {test_image}")
    else:
        print("没有找到测试图像")
        exit(1)

print(f"测试图像: {test_image}")
print(f"缓存目录: {adapter.depth_cache_dir}")

# 执行深度估计
try:
    print("开始深度估计...")
    depth = adapter.estimate_or_load_depth(test_image)
    print(f"深度估计完成: {depth.shape}, 范围: {depth.min():.3f}-{depth.max():.3f}")

    # 检查缓存
    cache_path = adapter.get_depth_cache_path(test_image)
    print(f"期望缓存路径: {cache_path}")

    if os.path.exists(cache_path):
        size = os.path.getsize(cache_path)
        print(f"✅ 缓存文件存在，大小: {size} bytes")
    else:
        print(f"❌ 缓存文件不存在")
        print("检查缓存目录内容:")
        if os.path.exists("./test_cache_direct"):
            files = os.listdir("./test_cache_direct")
            print(f"目录内容: {files}")

except Exception as e:
    print(f"错误: {e}")
    import traceback

    traceback.print_exc()