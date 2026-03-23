@echo off
setlocal enabledelayedexpansion

set "base_dir=save"

echo =================================================
echo   Starting Evaluation Pipeline...
echo =================================================

rem Iterate through folders inside ../save/
for /f "delims=" %%D in ('dir /b /ad "..\save\exp_*" 2^>nul') do (
    rem Construct the clean relative path that apls.py expects
    set "target_dir=!base_dir!/%%D"
    set "full_path=..\!base_dir!\%%D"
    
    echo.
    echo ========= Evaluating %%D =========
    
    if not exist "!full_path!\results\apls.json" (
        echo     -^> [RUNNING] Calculating APLS...
        call "%~dp0apls.cmd" "!target_dir!"
    ) else (
        echo     -^> [SKIPPED] APLS already exists.
    )

    if not exist "!full_path!\results\topo.json" (
        echo     -^> [RUNNING] Calculating TOPO...
        call "%~dp0topo.cmd" "!target_dir!"
    ) else (
        echo     -^> [SKIPPED] TOPO already exists.
    )
)

echo.
echo =================================================
echo   All evaluations finished!
echo   Please run 'python params_aggregate_rsts.py' in the root directory.
echo =================================================

endlocal
