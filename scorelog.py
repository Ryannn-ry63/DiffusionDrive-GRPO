from tensorboard.backend.event_processing import event_accumulator

log_path = '/inspire/hdd/global_user/wangcaojun-240208020180/nry/exp/tensorboard_logs/diffusiondrive/experiment_seed_0'
ea = event_accumulator.EventAccumulator(log_path)
ea.Reload()

# 打印出所有可用的标量标签
print("你的日志里包含以下标签：")
for tag in ea.Tags()['scalars']:
    print(f"- {tag}")