@echo off
setlocal enabledelayedexpansion

rem === 目录参数，去除多余引号 ===
set "dir=%~1"
set "data_dir=spacenet"

rem === 创建输出目录 ===
mkdir "..\!dir!\results\apls" >nul 2>&1

rem === 获取 test 图像名列表 ===
for /f "delims=" %%i in ('jq -r ".test[]" "..\spacenet\data_split.json"') do (
    set "name=%%i"
    rem %%%%%%%%%%%%
    set "gt_path=..\!data_dir!\RGB_1.0_meter\!name!__gt_graph.p"
    set "pred_path=..\!dir!\graph\!name!.p"

    rem 先用 pushd 进入当前目录，再用 cd 得到绝对路径
    pushd %cd%
    for %%P in ("!gt_path!") do set "abs_gt_path=%%~fP"
    for %%Q in ("!pred_path!") do set "abs_pred_path=%%~fQ"
    popd

    @REM echo Checking GT: !abs_gt_path!
    @REM echo Checking Prediction: !abs_pred_path!

    rem %%%%%%%%%%%%
    rem 用延迟变量展开来引用变量，避免路径拼接出错
    if exist "..\!dir!\graph\!name!.p" (
        echo ========================!name!======================

        rem 打印命令，便于调试
        @REM echo python apls\convert.py "..\!data_dir!\RGB_1.0_meter\!name!__gt_graph.p" gt.json
        @REM echo python apls\convert.py "..\!dir!\graph\!name!.p" prop.json

        rem 真正执行命令，注意用 !name! 而非 %%i，保持一致
        python apls\convert.py "..\!data_dir!\RGB_1.0_meter\!name!__gt_graph.p" gt.json
        python apls\convert.py "..\!dir!\graph\!name!.p" prop.json

        @REM 先push后pop，保证在apls路径下执行，避免因为go.mod和go.sum文件导致的错误
        pushd apls
        call "E:\Softwares\GO\go1.24.2.windows-amd64\go\bin\go.exe" run main.go ..\gt.json ..\prop.json "..\..\!dir!\results\apls\!name!.txt" spacenet
        popd

    )
)

python apls.py --dir "!dir!"

endlocal
