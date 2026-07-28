# world_model

pacman -S uv swig

uv python install 3.13

uv venv --python 3.13 .venv

. .venv/bin/activate

python -m ensurepip --upgrade

python -m pip install -r requirements.txt

python -m sampling.sample

python -m training.vae_train

python -m training.controller_train
