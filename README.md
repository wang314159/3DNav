# README
## 文件概览
### src/
用于进行单独测试的文件，训练时不会运行
 - elevationMap.cpp 点云转高程图测试文件
 - lidar.cpp lidar测试文件
 - rayDemo2.cpp 直接在raisim中控制车辆运动
### build/
src文件夹中编译出的可执行文件位置
### data/
数据文件夹，主要为urdf文件和地形图图片，地形图已废弃
### raisimGymTorch/
官方PPO代码会使用该文件夹存放数据，无用
### rl/
强化学习文件夹，基于raisim官方raisim_env_anymal修改得到
主要文件：
 - build_develop.sh 强化学习c++环境编译脚本，在rl文件夹下运行
 - build_debug.sh 强化学习c++环境编译脚本debug版，在rl文件夹下运行
 - raisimGymTorch 强化学习代码位置
#### rl/raisimGymTorch/
 - **algo** 基本算法代码文件夹
 - **env** 环境文件夹
    RaisimGymVecEnv.py为官方原版（PPO）环境，EleRLRaisimGymVecEnv.py为修改后环境
   
    目前所用环境在envs/nanocar/下，runner.py与tester.py为使用官方PPO算法的训练与测试代码，new_sac_runner.py与new_tester.py为SAC版本的训练与测试代码，manual_tester.py为使用lattice后的手动测试文件;cfg.yaml为参数文件，若要新增参数须在parameters.hpp中添加;Environment.hpp中为环境代码
    

## 环境配置
python所需环境可参考raisim官方raisimGymTorch所需环境，使用wandb记录数据，在205那台电脑上将conda环境切换到3DNav即可（建议直接导出）

c++须有OpenCV和PCL库

## 代码运行
rl文件夹下运行如下命令编译
```bash
source build_develop.sh
```

运行new_sac_runner.py进行训练，new_tester.py可读取pt文件进行测试
## Tips
建议主目录下新建raisim文件夹，包含raisimLib与raisimProject文件夹，官方库安装在raisimLib中，本仓库安装在raisimProject中

目前代码为使用了lattice生成轨迹的版本，学习效果并不好，可尝试修改网络或选择其他轨迹生成方式

环境与算法均有较大的可优化空间，如果想做完的话需要修改一下环境与机器人控制方式等

由于raisim为基于力的仿真器，因此在控制车轮速度时需用到pd控制器，目前使用的是raisim内置的控制器，但该控制器为基于位置的pd控制，并不是对速度的控制，因此效果并不好，后续可以尝试自行实现一个pid控制器
