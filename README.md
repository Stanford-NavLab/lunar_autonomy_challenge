# lunar_autonomy_challenge

Clone this repo inside the unzipped LunarAutonomyChallenge folder provided by the organizers which contains the simulator:

```
  LunarAutonomyChallenge
    ...
    lunar_autonomy_challenge
    ...
```

Create an `outputs/` folder to store generated data, and a `data/` folder to store other data (heightmaps, etc.).

## Conventions

### Transformations

`a_T_b` denotes the transformation from frame `b` to frame `a`.

- Also equivalent to the pose of frame `b` in frame `a`.
- `a_T_b * b_T_c = a_T_c`
- `a_T_b * b_P = a_P` (where `b_P` is points `P` in frame `b`)
