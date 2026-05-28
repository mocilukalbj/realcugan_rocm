@echo off
chcp 65001 >nul
title Clear cuDNN Cache

echo Clearing cuDNN benchmark cache...
if exist ".cudnn_benchmark_done" (
    del /f ".cudnn_benchmark_done"
    echo [OK] Deleted .cudnn_benchmark_done
) else (
    echo [OK] No cache file found
)

echo.
echo Cache cleared. Next run will re-benchmark.
pause