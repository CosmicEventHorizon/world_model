# world_model

pacman -S uv swig
uv python install 3.13
uv venv --python 3.13 .venv
. .venv/bin/activate
python -m ensurepip --upgrade
pip install -r requirements.txt