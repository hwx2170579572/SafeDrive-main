@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

REM ============================================================
REM LLM-guided RL midlevel formal pipeline runner
REM Put this .bat in the SAME folder as:
REM single_file_llm_guided_rl_v6_gpu_priority2_tensorboard_api_midlevel_final.py
REM ============================================================

cd /d "%~dp0"

set "PYTHON_FILE=single_file_llm_guided_rl_v6_gpu_priority2_tensorboard_api_midlevel_final.py"
set "RESULTS_DIR=results"
set "FREEZE_FILE=%RESULTS_DIR%\frozen_protocol_p2_real_api_midlevel.json"

if not exist "%PYTHON_FILE%" (
  echo [ERROR] Cannot find %PYTHON_FILE% in current folder:
  echo %cd%
  echo Please place this .bat in the same folder as the Python file.
  pause
  exit /b 1
)

if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%RESULTS_DIR%\tb_dev_ours_midlevel_smoke" mkdir "%RESULTS_DIR%\tb_dev_ours_midlevel_smoke"
if not exist "%RESULTS_DIR%\tb_formal_ours_midlevel_seed142" mkdir "%RESULTS_DIR%\tb_formal_ours_midlevel_seed142"
if not exist "%RESULTS_DIR%\tb_formal_compare_midlevel" mkdir "%RESULTS_DIR%\tb_formal_compare_midlevel"


echo.
echo ============================================================
echo [STEP 1/4] DEV smoke test with real LLM
echo ============================================================
echo.
python "%PYTHON_FILE%" ^
  --workflow-stage dev ^
  --allow-real-llm-smoke ^
  --mode constrained_real_llm_hier ^
  --episodes 5 ^
  --max-steps 200 ^
  --eval-episodes 2 ^
  --seed 42 ^
  --llm-backend real ^
  --tensorboard-dir %RESULTS_DIR%/tb_dev_ours_midlevel_smoke ^
  --llm-call-log-path %RESULTS_DIR%/dev_ours_midlevel_smoke_llm_calls.json ^
  --train-log-path %RESULTS_DIR%/dev_ours_midlevel_smoke_train_diag.csv ^
  --train-json-path %RESULTS_DIR%/dev_ours_midlevel_smoke_train_diag.json ^
  --train-plot-path %RESULTS_DIR%/dev_ours_midlevel_smoke_train_diag.png
if errorlevel 1 goto :FAILED


echo.
echo ============================================================
echo [STEP 2/4] Freeze new formal protocol
echo ============================================================
echo.
python "%PYTHON_FILE%" ^
  --workflow-stage freeze ^
  --freeze-save %FREEZE_FILE% ^
  --formal-modes baseline_sac,shaping_sac,rule_hier,constrained_real_llm_hier ^
  --formal-seeds 142,242,342 ^
  --min-successful-runs-per-mode 1 ^
  --episodes 150 ^
  --max-steps 200 ^
  --eval-episodes 8 ^
  --llm-backend real ^
  --llm-retry-times 3 ^
  --llm-retry-backoff 1,3,5 ^
  --lambda-ema-beta 0.97 ^
  --min-lambda-collision 0.01 ^
  --min-lambda-headway 0.01 ^
  --collision-risk-ttc-threshold 4.0 ^
  --collision-risk-front-distance 10.0
if errorlevel 1 goto :FAILED

if not exist "%FREEZE_FILE%" (
  echo [ERROR] Freeze protocol file was not created:
  echo %FREEZE_FILE%
  pause
  exit /b 1
)


echo.
echo ============================================================
echo [STEP 3/4] Formal single run ^(Ours, seed 142^)
echo ============================================================
echo.
python "%PYTHON_FILE%" ^
  --workflow-stage formal ^
  --freeze-load %FREEZE_FILE% ^
  --formal-strict ^
  --mode constrained_real_llm_hier ^
  --seed 142 ^
  --tensorboard-dir %RESULTS_DIR%/tb_formal_ours_midlevel_seed142 ^
  --llm-call-log-path %RESULTS_DIR%/formal_ours_midlevel_seed142_llm_calls.json ^
  --train-log-path %RESULTS_DIR%/formal_ours_midlevel_seed142_train_diag.csv ^
  --train-json-path %RESULTS_DIR%/formal_ours_midlevel_seed142_train_diag.json ^
  --train-plot-path %RESULTS_DIR%/formal_ours_midlevel_seed142_train_diag.png
if errorlevel 1 goto :FAILED


echo.
echo ============================================================
echo [STEP 4/4] Formal compare

echo ============================================================
echo.
python "%PYTHON_FILE%" ^
  --workflow-stage formal ^
  --freeze-load %FREEZE_FILE% ^
  --formal-strict ^
  --mode compare ^
  --tensorboard-dir %RESULTS_DIR%/tb_formal_compare_midlevel ^
  --compare-save-json %RESULTS_DIR%/formal_compare_midlevel.json ^
  --compare-save-csv %RESULTS_DIR%/formal_compare_midlevel.csv ^
  --compare-save-latex %RESULTS_DIR%/formal_compare_midlevel.tex
if errorlevel 1 goto :FAILED


echo.
echo ============================================================
echo All steps finished successfully.
echo ============================================================
echo.
echo Freeze file:
if exist "%FREEZE_FILE%" echo   %FREEZE_FILE%
echo.
echo TensorBoard examples:
echo   tensorboard --logdir %RESULTS_DIR%/tb_formal_ours_midlevel_seed142 --port 6006
echo   tensorboard --logdir %RESULTS_DIR%/tb_formal_compare_midlevel --port 6006
echo.
echo Open in browser:
echo   http://localhost:6006
pause
exit /b 0

:FAILED
echo.
echo ============================================================
echo Pipeline stopped because the previous command failed.
echo Please check the error above.
echo ============================================================
pause
exit /b 1
