TRAIN_TEST_SPLIT=navtest
#CHECKPOINT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/diffusiondrive_nusc_stage2.pth
"CHECKPOINT=/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/training_diffusiondrive_agent/2025.11.30.10.06.04/lightning_logs/version_0/checkpoints/model.ckpt"
python $NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score.py train_test_split=navtest agent=diffusiondrive_agent worker=ray_distributed agent.checkpoint_path=$CHECKPOINT experiment_name=diffusiondrive_agent_eval 
