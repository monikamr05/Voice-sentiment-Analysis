@echo off
cd /d "%~dp0"
call venv312\Scripts\activate
python train_model.py
pause
