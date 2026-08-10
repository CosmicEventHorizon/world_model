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

| Variant                    | Variable            | Average Reward Per 10 Rollouts |
| -------------------------- | ------------------- | ------------------------------ |
| Pixels only                | `x_t`               | TBD                            |
| Pixels + Velocity          | `x_t`, `v_t`        | TBD                            |
| Pixels + Memory            | `x_t`, `h_t`        | TBD                            |
| Pixels + Velocity + Memory | `x_t`, `v_t`, `h_t` | TBD                            |
| Latent only                | `z_t`               | TBD                            |
| Latent + Velocity          | `z_t`, `v_t`        | TBD                            |
| Latent + Memory            | `z_t`, `h_t`        | TBD                            |
| Latent + Velocity + Memory | `z_t`, `v_t`, `h_t` | TBD                            |
