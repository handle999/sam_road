@echo off
setlocal enabledelayedexpansion

:: Resolve the absolute path of the save directory
pushd ..\save
set "SAVE_DIR=%CD%"
popd

echo =================================================
echo   Starting Evaluation Pipeline...
echo =================================================

for /d %%D in ("%SAVE_DIR%\exp_*") do (
    echo.
    echo ========= Evaluating %%~nxD =========
    
    if not exist "%%D\apls_result.json" (
        echo     -^> [RUNNING] Calculating APLS...
        call apls.cmd "%%D"
    ) else (
        echo     -^> [SKIPPED] APLS already exists.
    )

    if not exist "%%D\topo_result.json" (
        echo     -^> [RUNNING] Calculating TOPO...
        call topo.cmd "%%D"
    ) else (
        echo     -^> [SKIPPED] TOPO already exists.
    )
)

echo.
echo =================================================
echo   All evaluations finished!
echo   Triggering Result Aggregator...
echo =================================================

:: Execute the aggregator script located in the parent directory
python ..\params_aggregate_rsts.py

endlocal