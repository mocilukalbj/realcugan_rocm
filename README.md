# Real-CUGAN ROCm 超分辨率工具

基于 [Real-CUGAN](https://github.com/Tohrusky/Real-CUGAN) 的超分辨率工具，针对 AMD ROCm GPU 优化。

## 特性

- **ROCm GPU 加速** - 支持 AMD 显卡 (RX 9070 XT, RX 7900 XTX 等)
- **异步流水线** - 多线程预读，减少 GPU 空闲时间
- **去重功能** - 自动跳过重复帧，显著减少工作量
- **批量处理** - 支持文件夹批量处理多个视频
- **多种模型** - 支持 Pro 和 V3 系列模型

## 环境要求

- Python 3.8+
- PyTorch (ROCm 版本)
- FFmpeg
- AMD GPU + ROCm 驱动

## 安装

```bash
# 激活虚拟环境
cd realcugan-rocm
venv\Scripts\activate

# 或使用已有的 Python 环境
pip install torch torchvision opencv-python numpy
```

## 快速开始

### 1. 配置文件

编辑 `配置_批处理.txt`：

```ini
# 输入输出目录
输入 = C:\Users\Administrator\Desktop\input
输出 = C:\Users\Administrator\Desktop\output

# 模型选择: pro-conservative-up2x, v3-conservative-up2x 等
模型 = pro-conservative-up2x
放大倍数 = 2

# 高级选项
分块模式 = 0    # 0=不分块, 2=2x2分块
批大小 = 4
去重 = 是       # 跳过重复帧
```

### 2. 运行

双击 `run_async_v3.bat` 或命令行运行：

```bash
python run_video_async.py --config 配置_批处理.txt
```

## 项目文件说明

| 文件 | 用途 |
|------|------|
| `run_video_async.py` | **主程序** - 异步流水线版本（推荐） |
| `run_video_batch.py` | 原版批量处理 |
| `run_video_batch_optimized.py` | 优化版批量处理 |
| `run_video_workerpool.py` | 多视频 Worker Pool |
| `配置_批处理.txt` | 配置文件 |

## 性能优化

### 异步流水线

```
┌─────────────────────────────────────────┐
│  主线程: GPU 处理帧 N                   │
│  同时后台线程: 读取帧 N+1               │
│  → GPU 几乎不等待 I/O                   │
└─────────────────────────────────────────┘
```

### 去重功能

动漫视频常有大量重复帧。开启去重后：
- 自动检测 64x64 缩略图相似的帧
- 直接复用上一帧的超分结果
- 显著减少 GPU 工作量

### 使用 Worker Pool (多视频)

```bash
python run_video_workerpool.py <输入文件夹> <输出文件夹> --parallel 4
```

## 模型说明

| 模型 | 特点 |
|------|------|
| `pro-conservative-up2x` | Pro 系列保守版，适合低质量源 |
| `pro-no-denoise-up2x` | Pro 系列无降噪，适合高质量源 |
| `v3-conservative-up2x` | V3 系列保守版 |
| `v3-no-denoise-up2x` | V3 系列无降噪 |

## 命令行参数

```bash
python run_video_async.py <input> <output> [options]

选项:
  --config CONFIG        配置文件路径
  --model MODEL          模型文件
  --scale SCALE          放大倍数 (2/3/4)
  --tile-mode MODE       分块模式 (0=不分块)
  --cache-mode MODE      缓存模式 (0-3)
  --batch-size SIZE      批处理大小
  --dedup                启用去重
  --fp32                 使用 FP32 (默认 FP16)
```

## 常见问题

### Q: 显存不够怎么办？
A: 减小 `批大小` 或启用 `分块模式 = 2`

### Q: 处理速度慢？
A: 1) 开启去重 2) 使用异步版本 3) 减小 tile_mode

### Q: 有拼接缝隙？
A: 将 `分块模式` 设为 0，但这会增加显存占用

## 技术细节

- 异步流水线使用 `threading.Queue` 实现双缓冲
- 去重使用 64x64 缩略图比较，阈值 3.0
- 预热后模型常驻 GPU，避免重复加载

## 许可证

MIT License - 仅供学习交流使用。

## 参考

- [Real-CUGAN](https://github.com/Tohrusky/Real-CUGAN)
- [PyTorch ROCm](https://pytorch.org/)
- [FFmpeg](https://ffmpeg.org/)