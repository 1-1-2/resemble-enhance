@echo off

python -m venv venv
call venv/scripts/activate

REM pip install -v torch==2.3.1+cu118 torchaudio==2.3.1+cu118 -f https://mirror.sjtu.edu.cn/pytorch-wheels/cu118
pip install -v torch==2.3.1+cu118 torchaudio==2.3.1+cu118 -f https://mirrors.aliyun.com/pytorch-wheels/cu118
pip install -r requirements.txt

@echo Instaling deepspeed for python 3.10.x or 3.11 and CUDA 11.8
python deepspeed_installer.py

@echo Install complete
pause