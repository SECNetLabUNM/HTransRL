# Hybrid-Transformer based Multi-agent Reinforcement Learning for Multiple Unmanned Aerial Vehicle Coordination in Air Corridors
## Animation
#### cttc, one-transfer
4 air corridors, cylinder-torus-torus-cylinder, 12 UAVs, 4-static, and 3-moving

![cttc_12.gif](test%20and%20visualization%2Fmd_present%2Fanimation%2Fcttc_12.gif)
#### cttcttcttc, 3-transfer
10 air corridors, cylinder-torus-torus-cylinder-torus-torus-cylinder-torus-torus-cylinder, 12 UAVs, 4-static, and 3-moving

![cttcttcttc_12.gif](test%20and%20visualization%2Fmd_present%2Fanimation%2Fcttcttcttc_12.gif)


## Modeling
### Air Corridor, Cylinder and Torus
![Air_corridor.jpg](test%20and%20visualization%2Fmd_present%2FAir_corridor.jpg)

## RL Training
### Network Structure
- H(), embedding layer, normalizes the input values and standardize the input dimensions.
- G(), transformer layer, deals with stochastic neighbors information
- F(), actor-critic network combined.
![TransRL.jpg](test%20and%20visualization%2Fmd_present%2FHTransRL.jpg)



### Training File
#### One set of parameters: [main.py](rl_multi_3d_trans%2Fmain.py)
#### Batch process, for parameter grid search: [batched_grid_search.sh](rl_multi_3d_trans%2Fbatched_grid_search.sh)
Models (actor/critic) are saved every 0.25 million steps
Training steps can be visualized with terminal log and TensorBoard.

### Test File
#### Series, generate animation: [D3MOVE_test_single_core.py](test%20and%20visualization%2FD3MOVE_test_single_core.py)
#### Parallel, generate data for figs: [D3MOVE_test_parallel.py](test%20and%20visualization%2FD3MOVE_test_parallel.py)