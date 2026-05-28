"""
Real-CUGAN 异步流水线版 - 单帧处理 + 异步 I/O 减少 GPU 空闲

核心优化:
1. 后台线程预读下一帧原始字节，与当前帧 GPU 推理重叠
2. 预热后保持模型常驻 GPU，避免重复加载

用法: python run_video_async.py <输入> <输出> [--dedup]
"""
import os
import sys
import atexit
import time
import argparse
import subprocess
import shutil
import threading
import queue
import signal
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch

atexit.register(torch.cuda.empty_cache)

_shutdown_flag = False
_cleanup_procs = []


def _oom_handler():
    global _shutdown_flag
    _shutdown_flag = True
    for proc_in, proc_out in _cleanup_procs:
        try:
            proc_in.stdout.close()
        except Exception:
            pass
        try:
            proc_out.stdin.close()
        except Exception:
            pass
    print("\n[OOM] Out of VRAM — shutting down cleanly", flush=True)
    sys.exit(1)


def _signal_handler(signum, frame):
    _oom_handler()


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128")

sys.path.insert(0, str(Path(__file__).parent))
from upcunet_v3 import RealWaifuUpScaler, _mark_cudnn_benchmark_done, _mark_compile_done

MODEL_DIR = Path(__file__).parent
SUPPORTED_VIDEO = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v'}


def _find_ffmpeg():
    for name in ["ffmpeg", "ffmpeg.exe"]:
        if shutil.which(name):
            ffmpeg = shutil.which(name)
            base = str(Path(ffmpeg).parent)
            ffprobe = os.path.join(base, "ffprobe.exe" if os.name == "nt" else "ffprobe")
            if not os.path.exists(ffprobe):
                ffprobe = shutil.which("ffprobe") or ""
            return ffmpeg, ffprobe if os.path.exists(ffprobe) else ""
    return None, None


def _probe(ffmpeg, ffprobe, path):
    """探测视频信息，增加调试输出"""
    import json
    import re
    info = {"video": [], "audio": [], "subtitle": [], "duration": 0}

    print(f"[Debug] Probing: {path}")
    print(f"[Debug] FFprobe: {ffprobe}")

    if ffprobe and Path(ffprobe).exists():
        try:
            r = subprocess.run(
                [ffprobe, "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-show_format", str(path)],
                capture_output=True, text=True, timeout=30)
            print(f"[Debug] FFprobe return: {r.returncode}")
            if r.returncode == 0:
                data = json.loads(r.stdout)
                for s in data.get("streams", []):
                    kind = s.get("codec_type", "")
                    entry = {"index": s.get("index", 0),
                             "codec": s.get("codec_name", ""),
                             "language": s.get("tags", {}).get("language", "")}
                    if kind == "video":
                        entry["width"] = s.get("width", 0)
                        entry["height"] = s.get("height", 0)
                        p = s.get("r_frame_rate", "30/1").split("/")
                        entry["fps"] = float(p[0]) / float(p[1]) if p[1] != "0" else 30
                        info["video"].append(entry)
                    elif kind == "audio":
                        info["audio"].append(entry)
                    elif kind == "subtitle":
                        info["subtitle"].append(entry)
                info["duration"] = float(data.get("format", {}).get("duration", 0))
                print(f"[Debug] Found {len(info['video'])} video streams")
                return info
        except Exception as e:
            print(f"[Debug] FFprobe error: {e}")
            pass
    # Fallback: use ffmpeg directly
    try:
        r = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True, timeout=30)
        s = r.stderr
        d = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", s)
        if d:
            info["duration"] = int(d[1])*3600 + int(d[2])*60 + int(d[3]) + int(d[4])/100
        for m in re.finditer(r"Stream #\d+:(\d+)(?:\[.+?\])?(?:\((\w+)\))?:\s*(\w+):\s*(.+)", s):
            idx, lang, kind, details = int(m[1]), m[2] or "", m[3].lower(), m[4]
            entry = {"index": idx, "codec": "", "language": lang}
            cm = re.match(r"(\S+)", details)
            if cm:
                entry["codec"] = cm[1]
            if kind == "video":
                r2 = re.search(r"(?<!\w)(\d{2,5})x(\d{2,5})(?=\W|$)", details)
                if r2:
                    entry["width"], entry["height"] = int(r2[1]), int(r2[2])
                f2 = re.search(r"([\d.]+)\s*fps", details)
                entry["fps"] = float(f2[1]) if f2 else 30
                info["video"].append(entry)
            elif kind == "audio":
                info["audio"].append(entry)
            elif kind == "subtitle":
                info["subtitle"].append(entry)
    except Exception:
        pass
    return info


def _read_exact(pipe, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(min(n - len(buf), 1024 * 1024))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


class AsyncPipeline:
    """异步流水线：GPU 处理当前帧时，后台线程预读下一帧"""

    def __init__(self, upscaler, args):
        self.upscaler = upscaler
        self.args = args
        self.model = upscaler.model
        self.device = upscaler.device
        self.pro = upscaler.pro

        # 双缓冲队列：生产者(读帧) → 消费者(GPU处理)
        self.frame_queue = queue.Queue(maxsize=2)
        self.read_done = False
        self.read_thread = None

        # 去重缓存
        dedup_window = args.dedup_window or max(4, 24)
        self.recent_frames = deque(maxlen=dedup_window)
        self.DEDUP_THRESHOLD = 3.0
        self.dedup_lock = threading.Lock()

    def process_video(self, inp_path, opt_path):
        """异步流水线处理视频"""
        args = self.args
        ffmpeg, ffprobe = _find_ffmpeg()
        print(f"[Debug] FFmpeg: {ffmpeg}, FFprobe: {ffprobe}")

        if not ffmpeg:
            return {"error": "FFmpeg not found"}

        info = _probe(ffmpeg, ffprobe, inp_path)
        if not info["video"]:
            return {"error": "No video stream"}

        s = info["video"][0]
        width, height = s["width"], s["height"]
        fps = s["fps"]
        new_w, new_h = width * args.scale, height * args.scale
        frame_size = width * height * 3

        # 解码命令
        decode_cmd = [ffmpeg, "-y", "-i", str(inp_path),
                      "-f", "rawvideo", "-pix_fmt", "bgr24",
                      "-vcodec", "rawvideo", "-an", "-sn", "pipe:1"]

        # 编码命令
        encode_cmd = [ffmpeg, "-y",
                      "-f", "rawvideo", "-pix_fmt", "bgr24",
                      "-s", f"{new_w}x{new_h}", "-r", str(fps),
                      "-i", "pipe:0", "-i", str(inp_path),
                      "-map", "0:v", "-map", "1:a?"]
        for a in info["audio"]:
            encode_cmd += ["-map", f"1:{a['index']}"]
        encode_cmd += ["-c:a", "aac", "-b:a", "192k"] if info["audio"] else []
        for sub in info["subtitle"]:
            encode_cmd += ["-map", f"1:{sub['index']}"]
        encode_cmd += ["-c:s", "mov_text" if opt_path.suffix == '.mp4' else "copy"] if info["subtitle"] else []

        # 检测编码器
        try:
            enc_r = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, timeout=10)
            encoders = enc_r.stdout
        except Exception:
            encoders = ""

        if " hevc_amf " in encoders:
            encode_cmd += ["-c:v", "hevc_amf", "-quality", "quality", "-qp_i", str(args.amf_qp), "-qp_p", str(args.amf_qp + 2)]
        elif " h264_amf " in encoders:
            encode_cmd += ["-c:v", "h264_amf", "-quality", "quality", "-qp_i", str(args.amf_qp), "-qp_p", str(args.amf_qp + 2)]
        else:
            encode_cmd += ["-c:v", "libx265", "-crf", str(args.crf)]

        encode_cmd += ["-pix_fmt", "yuv420p", str(opt_path)]

        print(f"\n[Input]  {inp_path.name}  |  {width}x{height} → {new_w}x{new_h}  |  {fps:.2f}fps")
        print(f"[Async]  Pipeline enabled (read+dedup overlap with GPU)\n")

        proc_in = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        proc_out = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

        _cleanup_procs.append((proc_in, proc_out))

        try:
            self._run_loop(proc_in, proc_out, width, height, frame_size, info, fps)
        finally:
            _cleanup_procs.remove((proc_in, proc_out))
            try:
                proc_in.stdout.close()
            except Exception:
                pass
            try:
                proc_out.stdin.close()
            except Exception:
                pass
            proc_in.wait()
            proc_out.wait()

        return {"success": True, "frames": self.frame_idx}

    def _run_loop(self, proc_in, proc_out, width, height, frame_size, info, fps):
        args = self.args
        tile_mode = args.tile_mode
        cache_mode = args.cache_mode
        alpha = args.alpha
        pro = self.pro

        # ── 预热 ──
        print(f"[GPU]     Warmup (JIT compile)...", flush=True)
        warm_raw = _read_exact(proc_in.stdout, frame_size)
        if len(warm_raw) != frame_size:
            return

        warm_bgr = np.frombuffer(warm_raw, dtype=np.uint8).reshape(height, width, 3)
        warm_rgb = warm_bgr[:, :, [2, 1, 0]]

        with torch.no_grad():
            _ = self.upscaler(warm_rgb, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)
        torch.cuda.synchronize()

        # 写回 warmup 帧
        warm_result = self.upscaler(warm_rgb, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)
        warm_bgr_out = np.ascontiguousarray(warm_result[:, :, ::-1])
        proc_out.stdin.write(warm_bgr_out.tobytes())

        if args.dedup:
            self.recent_frames.append((cv2.resize(warm_bgr, (64, 64)), warm_bgr_out.tobytes()))

        self.frame_idx = 1
        print(f"[GPU]     Warmup done, starting async pipeline...\n", flush=True)

        # 标记 benchmark 和 compile 完成，创建缓存文件
        _mark_cudnn_benchmark_done()
        _mark_compile_done()

        # ── 启动异步预读线程 ──
        self.frame_queue = queue.Queue(maxsize=2)  # 双缓冲
        self.read_done = False
        self.read_error = None

        def background_reader():
            """后台线程：不断读取帧并放入队列"""
            try:
                while True:
                    raw = _read_exact(proc_in.stdout, frame_size)
                    if len(raw) != frame_size:
                        break
                    # 阻塞直到队列有空位
                    self.frame_queue.put(raw, block=True, timeout=30)
                self.frame_queue.put(None)  # 发送结束信号
            except Exception as e:
                self.read_error = str(e)
                self.frame_queue.put(None)

        self.read_thread = threading.Thread(target=background_reader, daemon=True)
        self.read_thread.start()

        # 等待第一帧数据
        pending_read = self.frame_queue.get(block=True, timeout=60)

        total_est = int(info.get("duration", 0) * fps) if info.get("duration", 0) > 0 else None
        dup_count = 0
        total_start = time.time()
        last_print = 0

        while pending_read is not None and len(pending_read) == frame_size:
            # ── 1. 处理当前帧（已预读） ──
            cur_bgr = np.frombuffer(pending_read, dtype=np.uint8).reshape(height, width, 3)

            # 去重检查
            is_dup = False
            dup_bytes = None
            if args.dedup:
                cur_thumb = cv2.resize(cur_bgr, (64, 64))
                for rt, rb in self.recent_frames:
                    if np.mean(np.abs(cur_thumb.astype(float) - rt.astype(float))) <= self.DEDUP_THRESHOLD:
                        is_dup = True
                        dup_bytes = rb
                        break

            tg0 = time.time()
            if not is_dup:
                cur_rgb = cur_bgr[:, :, [2, 1, 0]]
                result = self.upscaler(cur_rgb, tile_mode=tile_mode, cache_mode=cache_mode, alpha=alpha)
                torch.cuda.synchronize()

                out_bgr = np.ascontiguousarray(result[:, :, ::-1])
                out_bytes = out_bgr.tobytes()

                if args.dedup:
                    self.recent_frames.append((cur_thumb, out_bytes))
            else:
                out_bytes = dup_bytes
                dup_count += 1

            tg1 = time.time()

            # 写入编码器
            try:
                proc_out.stdin.write(out_bytes)
            except BrokenPipeError:
                break

            # ── 2. 从队列获取下一帧（后台线程已在并行预读） ──
            try:
                pending_read = self.frame_queue.get(block=True, timeout=30)
            except queue.Empty:
                print(f"\n[Warning] Frame read timeout")
                break

            self.frame_idx += 1

            # 进度显示
            now = time.time()
            if now - last_print > 1.0 or pending_read is None or len(pending_read) != frame_size:
                elapsed = now - total_start
                fps_p = self.frame_idx / elapsed if elapsed > 0 else 0
                pct = self.frame_idx / total_est * 100 if total_est else 0
                eta = "?" if not total_est else f"{(total_est - self.frame_idx) / fps_p / 60:.1f}m" if fps_p > 0 else "?"
                print(f"  [{self.frame_idx:5d}/{total_est or '?'}] {pct:5.1f}%  {fps_p:.1f}fps  ETA {eta}", end='\r', flush=True)
                last_print = now

        # 等待读取线程结束
        self.read_thread.join(timeout=5)

        elapsed = time.time() - total_start
        avg_fps = self.frame_idx / elapsed if elapsed > 0 else 0
        print(f"\n\n[Done] {self.frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} fps)")
        if dup_count > 0:
            print(f"  Dedup: {dup_count} duplicates skipped ({dup_count/self.frame_idx*100:.1f}%)")


def setup_model(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wpath = MODEL_DIR / args.model
    if not wpath.exists():
        sys.exit(f"[Error] Model not found: {wpath}")
    compile_enabled = getattr(args, 'compile', True)  # 默认启用
    return RealWaifuUpScaler(args.scale, str(wpath), half=not args.fp32, device=device, compile_enabled=compile_enabled)


def parse_config(path):
    cfg = {}
    if not Path(path).exists():
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and value:
                    cfg[key] = value

    alias_map = {
        "输入": "input", "输出": "output", "模式": "mode", "模型": "model",
        "放大倍数": "scale", "分块模式": "tile_mode",
        "AMF质量": "amf_qp", "输出后缀": "suffix", "日志": "log",
        "去重": "dedup", "编译": "compile",
    }
    for cn, en in alias_map.items():
        if cn in cfg:
            cfg[en] = cfg[cn]
    return cfg


def main():
    p = argparse.ArgumentParser(description="Real-CUGAN Async Pipeline")
    p.add_argument("input", type=str, nargs="?")
    p.add_argument("output", type=str, nargs="?")
    p.add_argument("--config", type=str, default="")
    p.add_argument("--model", type=str, default="weights_pro/pro-conservative-up2x.pth")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--tile-mode", type=int, default=0)
    p.add_argument("--cache-mode", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--amf-qp", type=int, default=18)
    p.add_argument("--suffix", type=str, default="")
    p.add_argument("--dedup", action="store_true")
    p.add_argument("--dedup-window", type=int, default=0)
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no-compile", action="store_false", dest="compile")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # 读取配置文件
    config_path = args.config
    if config_path and Path(config_path).exists():
        cfg = parse_config(config_path)
        print(f"[Config] Read config: {config_path}")

        # 从配置文件覆盖参数（支持中英文key）
        if ("输入" in cfg or "input" in cfg) and not args.input:
            args.input = cfg.get("输入") or cfg.get("input")
        if ("输出" in cfg or "output" in cfg) and not args.output:
            args.output = cfg.get("输出") or cfg.get("output")
        if "模型" in cfg or "model" in cfg:
            model_map = {
                "pro-conservative-up2x": ("weights_pro/pro-conservative-up2x.pth", 2),
                "pro-no-denoise-up2x": ("weights_pro/pro-no-denoise-up2x.pth", 2),
                "pro-denoise3x-up2x": ("weights_pro/pro-denoise3x-up2x.pth", 2),
                "v3-conservative-up2x": ("weights_v3/up2x-latest-conservative.pth", 2),
                "v3-no-denoise-up2x": ("weights_v3/up2x-latest-no-denoise.pth", 2),
                "v3-denoise1x-up2x": ("weights_v3/up2x-latest-denoise1x.pth", 2),
                "v3-denoise2x-up2x": ("weights_v3/up2x-latest-denoise2x.pth", 2),
                "v3-denoise3x-up2x": ("weights_v3/up2x-latest-denoise3x.pth", 2),
            }
            model_name = cfg.get("模型") or cfg.get("model") or "pro-conservative-up2x"
            if model_name in model_map:
                args.model, default_scale = model_map[model_name]
                if str(args.scale) == "2" and default_scale != 2:
                    args.scale = default_scale
        if "放大倍数" in cfg or "scale" in cfg:
            args.scale = int(cfg.get("放大倍数") or cfg.get("scale") or 2)
        if "分块模式" in cfg or "tile_mode" in cfg:
            args.tile_mode = int(cfg.get("分块模式") or cfg.get("tile_mode") or 0)
        if "AMF质量" in cfg or "amf_qp" in cfg:
            args.amf_qp = int(cfg.get("AMF质量") or cfg.get("amf_qp") or 18)
        if "输出后缀" in cfg or "suffix" in cfg:
            args.suffix = cfg.get("输出后缀") or cfg.get("suffix") or ""
        log_val = cfg.get("日志") or cfg.get("log", "")
        if log_val.strip().lower() in ("是", "yes", "true", "1"):
            args.log = "log.txt"
        dedup_val = cfg.get("去重") or cfg.get("dedup", "")
        if dedup_val.strip().lower() in ("是", "yes", "true", "1"):
            args.dedup = True
        compile_val = cfg.get("编译") or cfg.get("compile", "")
        if compile_val.strip().lower() in ("否", "no", "false", "0"):
            args.compile = False

        print(f"[Config] Input:  {cfg.get('输入') or cfg.get('input') or 'N/A'}")
        print(f"[Config] Output: {cfg.get('输出') or cfg.get('output') or 'N/A'}")
        print(f"[Config] Model:  {cfg.get('模型') or cfg.get('model') or 'pro-conservative-up2x'}")
        print(f"[Config] Scale:  {cfg.get('放大倍数') or cfg.get('scale') or '2'}x")
        print(f"[Config] Tile:   {cfg.get('分块模式') or cfg.get('tile_mode') or '0'}")
        print(f"[Config] Dedup:  {cfg.get('去重') or cfg.get('dedup') or 'No'}")
        print(f"[Config] Compile: {cfg.get('编译') or cfg.get('compile') or 'Yes'}")
        print()

    if not args.input or not args.output:
        sys.exit("[Error] 需要指定输入输出路径或 --config")

    upscaler = setup_model(args)
    if torch.cuda.is_available():
        print(f"[Device] {torch.cuda.get_device_name(0)}")

    inp = Path(args.input)
    out = Path(args.output)
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    # 支持文件夹（批量处理多个视频）
    if inp.is_dir():
        vids = sorted([f for f in inp.iterdir()
                       if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO])
        if not vids:
            sys.exit(f"[Error] 文件夹中没有视频文件: {inp}")
        print(f"[Folder] 发现 {len(vids)} 个视频文件")
        for i, v in enumerate(vids, 1):
            out_path = out / f"{v.stem}{args.suffix or ''}{v.suffix}"
            print(f"\n[{i}/{len(vids)}] 处理: {v.name}")
            pipeline = AsyncPipeline(upscaler, args)
            result = pipeline.process_video(v, out_path)
            if result.get("error"):
                print(f"[Error] {v.name}: {result['error']}")
        print(f"\n[All Done] 处理完成 {len(vids)} 个视频")
    elif inp.is_file():
        # 单文件处理
        if out.is_dir():
            out = out / f"{inp.stem}{args.suffix or ''}{inp.suffix}"
        pipeline = AsyncPipeline(upscaler, args)
        result = pipeline.process_video(inp, out)
        if result.get("error"):
            print(f"[Error] {result['error']}")


if __name__ == "__main__":
    main()