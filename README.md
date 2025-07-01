This repository contains the code for the Stanford NAV Lab's official solution for the 2025 [Lunar Autonomy Challenge](https://lunar-autonomy-challenge.jhuapl.edu/).

## Setup

Clone this repo inside the unzipped LunarAutonomyChallenge folder provided by the organizers which contains the simulator:

```
  LunarAutonomyChallenge
    ...
    lunar_autonomy_challenge
    ...
```

Create an `outputs/` folder to store generated data, and a `data/` folder to store other data (heightmaps, etc.).

### Environment

1. Create conda env
```
conda create -n lac python=3.10
conda activate lac
```
2. Setup LAC simulator
- Download simulator folder from LunarAutonomyChallenge.zip 
- Unzip it into ~/ or desired location
- `cd ~/LunarAutonomyChallenge`
- `pip3 install --force-reinstall -r requirements.txt`
  
3. Install pytorch into lac environment: `pip3 install torch torchvision torchaudio`
4. Clone this repo into the LunarAutonomyChallenge folder. Inside ~/LunarAutonomyChallenge/lunar_autonomy_challenge:
```
pip install -r requirements.txt
pip install -e .
```
5. Install LightGlue. In `~/opt`:
```
git clone https://github.com/cvg/LightGlue.git && cd LightGlue
python -m pip install -e .
```
6. `pip install -U segmentation-models-pytorch`



## Conventions

### Transformations

We use the [GTSAM convention](https://gtsam.org/gtsam.org/2020/06/28/gtsam-conventions.html), where `a_T_b` denotes the transformation from frame `b` to frame `a`.

- Also equivalent to the pose of frame `b` in frame `a`.
- `a_T_b * b_T_c = a_T_c`
- `a_T_b * b_P = a_P` (where `b_P` is points `P` in frame `b`)
