```sh
uv python install 3.12
source .venv-py312/bin/activate
uv pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 -i https://mirrors.aliyun.com/pypi/simple/
uv pip install omnivoice-triton>=0.1.0 # 


```
