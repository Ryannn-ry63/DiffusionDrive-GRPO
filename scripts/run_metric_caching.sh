#!/bin/bash

# Metric Caching for navtest
# 用法: bash scripts/run_metric_caching.sh

cd /inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive

# 设置环境变量
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/global_public/public_datas/NAVSIM/maps"
export NAVSIM_EXP_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp"
export OPENSCENE_DATA_ROOT="/inspire/hdd/global_user/wangcaojun-240208020180/nry/dataset"
export PYTHONPATH="/inspire/hdd/global_user/wangcaojun-240208020180/nry/DiffusionDrive"

# 运行 Metric Caching
python navsim/planning/script/run_metric_caching.py \
    train_test_split=navtrain \
    cache.cache_path=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/metric_cache \
    cache.force_feature_computation=false
