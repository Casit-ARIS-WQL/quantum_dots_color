This is a test project!

## Color Detector - 黄色/白色颜色判别算法

基于 OpenCV 的颜色判别算法，通过参考图像自动校准黄色阈值，批量处理图像并判断指定区域为黄色或白色。

### 原理

1. **校准阶段**：提供一张包含目标黄色的参考图像，算法将其转换到 HSV 色彩空间，统计黄色像素的色调、饱和度和明度范围。
2. **检测阶段**：对每张待检测图像的指定区域（ROI），计算黄色像素占比。若超过设定阈值则判断为黄色，否则为白色。

### 安装

```bash
pip install -r requirements.txt
```

### 使用方法

#### 命令行

```bash
# 使用参考图像校准，批量处理目录中的图像
python color_detector.py --reference ref_image.png --input-dir ./images

# 指定 ROI 区域（x,y,宽,高）
python color_detector.py --reference ref.png --input-dir ./images \
    --ref-roi 100,100,50,50 --detect-roi 200,200,80,80

# 处理指定图像列表，自定义阈值
python color_detector.py --reference ref.png \
    --images img1.png img2.png img3.png --threshold 0.25
```

#### Python API

```python
from color_detector import ColorDetector

# 初始化检测器
detector = ColorDetector(yellow_ratio_threshold=0.3)

# 用参考图像校准黄色阈值
detector.calibrate("reference_yellow.png")

# 单张图像检测
result = detector.detect("test_image.png", roi=(100, 100, 50, 50))
print(result)  # {'image': '...', 'color': 'yellow', 'yellow_ratio': 0.45, ...}

# 批量处理目录
results = detector.batch_detect_from_directory("./images", roi=(100, 100, 50, 50))
for r in results:
    print(f"{r['image']}: {r['color']} (ratio: {r['yellow_ratio']})")
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `yellow_ratio_threshold` | 0.3 | 黄色像素占比阈值，超过则判定为黄色 |
| `hsv_margin` | 10 | HSV 范围扩展余量，用于适应光照变化 |
| `roi` | None | 感兴趣区域 (x, y, width, height)，None 表示整张图像 |
