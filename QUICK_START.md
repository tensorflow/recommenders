# 1. prepare venv
```
# init venv
mkdir .python-builds && cd .python-builds
wget https://github.com/astral-sh/python-build-standalone/releases/download/20260325/cpython-3.10.20%2B20260325-x86_64-unknown-linux-gnu-install_only.tar.gz
tar -zxvf cpython-3.10.20+20260325-x86_64-unknown-linux-gnu-install_only.tar.gz
cd ..
./.python-builds/python/bin/python3.10 -m venv ./.venv310

# start venv
source .venv310/bin/activate
python -m pip install -U pip setuptools wheel ipykernel
python -m ipykernel install --user --name py310-ml --display-name "Python (py310-ml)"

# exit venv
deactivate
```