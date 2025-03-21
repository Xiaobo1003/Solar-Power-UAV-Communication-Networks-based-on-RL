# Solar-Power-UAV-Communication-Networks-based-on-RL
This repository contains the official implementation for our paper:

**[Optimal Charging Profile Design for Solar-Powered Sustainable UAV Communication Networks](https://ieeexplore.ieee.org/abstract/document/10279806)**  
_Longxin Wang, Saugat Tripathi, Ran Zhang, Nan Cheng, Miao Wang_  
Published in *IEEE International Conference on Communication(ICC)*, 2023.

## 🖼️ Network Architecture
![UAV Network](UAV_System.png)

## 📁 Project Structure
- `DDPG 17UAV.py` - main script to run the 17 UAVs
- `UAVsolar17test` - Environment Setup
- `DDPG UAV15best.py` - main script to run the 15 UAVs
- `UAVsolar15test` - Environment Setup

## 📊 Numerical Results
### 🔹 Solar radiation dynamics
![Solar_radiation](Solar_radiation.png)
### 🔹 User demand dynamics
![User_demand_dynamics](User_demand_dynamics.png)
### 🔹 Convergence
![Convergence](Convergence.png)
### 🔹 Hourly number of serving UAVs
![Hourly number of serving UAVs](Hourly_number_of_serving_UAVs.png)


## 🚀 Usage

### ▶️ Run with 17 UAVs:

```bash
python "DDPG 17UAV.py"
```
### ▶️ Run with 15 UAVs:
```bash
python "DDPG UAV15best.py"
```
