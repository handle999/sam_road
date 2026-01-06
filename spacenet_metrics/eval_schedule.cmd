@echo off
setlocal enabledelayedexpansion

rem === 设置输出目录列表 ===
set output_dirs=sam_road_official_ep20

rem === 设置基准路径 ===
set base_dir=save

for %%D in (%output_dirs%) do (
    set "full_path=%base_dir%\%%D"
    echo.
    echo ========= Evaluating %%D =========
    call "%~dp0apls.cmd" "!full_path!"
    call "%~dp0topo.cmd" "!full_path!"
)

endlocal
