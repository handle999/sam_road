@echo off
setlocal enabledelayedexpansion

REM === 接收保存目录参数 ===
set "dir=%~1"
set "script_dir=%~dp0"

REM === 创建输出目录（可选，若你的topo有输出结果）===
mkdir "..\!dir!\results\topo" >nul 2>&1

REM === 打印调试信息 ===
echo Evaluating TOPO for !dir!

REM === 执行 topo/main.py ===
python "%script_dir%topo\main.py" -savedir "!dir!"

REM === 如有额外的topo.py逻辑，可解注此行 ===
python "%script_dir%topo.py" -savedir "!dir!"

endlocal
