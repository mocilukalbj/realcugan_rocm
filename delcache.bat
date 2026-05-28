@echo off
chcp 65001 >nul
title Clear Cache

echo Clearing cuDNN benchmark cache...
if exist ".cudnn_benchmark_done" (
    del /f ".cudnn_benchmark_done"
    echo [OK] Deleted .cudnn_benchmark_done
) else (
    echo [OK] No cuDNN cache file found
)

echo.
echo Clearing torch.compile cache...
if exist ".compile_cache_done" (
    del /f ".compile_cache_done"
    echo [OK] Deleted .compile_cache_done
) else (
    echo [OK] No compile cache file found
)

echo.
echo All caches cleared. Next run will re-benchmark and re-compile.
pause