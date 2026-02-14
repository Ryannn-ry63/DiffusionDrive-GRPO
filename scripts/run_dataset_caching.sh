#!/bin/bash

cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive

# 设置环境变量
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/global_public/public_datas/NAVSIM/maps"
export NAVSIM_EXP_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp"
export OPENSCENE_DATA_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset"
export PYTHONPATH="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"

# 运行 Dataset Caching
python navsim/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_agent \
    experiment_name=training_diffusiondrive_agent \
    train_test_split=navtrain \
    cache_path=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_cache \
    force_cache_computation=false
