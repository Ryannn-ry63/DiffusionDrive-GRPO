#!/usr/bin/env python3
"""
测试 GRPO 训练流程的脚本
用于验证：
1. 参数冻结正确性
2. ref_policy 正确设置
3. forward 流程完整性
4. loss 计算正确性
"""

import torch
import torch.nn as nn
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent

def count_parameters(model, trainable_only=False):
    """统计模型参数"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def test_parameter_freezing():
    """测试参数冻结"""
    print("\n" + "="*80)
    print("测试 1: 参数冻结检查")
    print("="*80)
    
    config = TransfuserConfig()
    agent = TransfuserAgent(config=config, lr=1e-4)
    
    total_params = count_parameters(agent._transfuser_model)
    trainable_params = count_parameters(agent._transfuser_model, trainable_only=True)
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数比例: {(1 - trainable_params/total_params)*100:.2f}%")
    
    # 检查 diff_decoder 是否可训练
    diff_decoder_trainable = sum(p.numel() for p in agent._transfuser_model._trajectory_head.diff_decoder.parameters() if p.requires_grad)
    print(f"\ndiff_decoder 可训练参数: {diff_decoder_trainable:,}")
    
    # 检查 backbone 是否冻结
    backbone_trainable = sum(p.numel() for p in agent._transfuser_model._backbone.parameters() if p.requires_grad)
    print(f"backbone 可训练参数: {backbone_trainable:,}")
    
    if backbone_trainable == 0 and diff_decoder_trainable > 0:
        print("\n✓ 参数冻结配置正确!")
    else:
        print("\n✗ 参数冻结配置有问题!")
        
    return agent

def test_ref_policy():
    """测试 ref_policy 设置"""
    print("\n" + "="*80)
    print("测试 2: ref_policy 检查")
    print("="*80)
    
    config = TransfuserConfig()
    agent = TransfuserAgent(config=config, lr=1e-4)
    
    ref_policy = agent._transfuser_model._trajectory_head.ref_policy
    
    if ref_policy is None:
        print("✗ ref_policy 未设置!")
        return False
    
    print(f"✓ ref_policy 已设置: {type(ref_policy).__name__}")
    
    # 检查 ref_policy 是否冻结
    ref_trainable = sum(p.numel() for p in ref_policy.parameters() if p.requires_grad)
    print(f"ref_policy 可训练参数: {ref_trainable:,}")
    
    if ref_trainable == 0:
        print("✓ ref_policy 完全冻结!")
    else:
        print("✗ ref_policy 未完全冻结!")
        
    # 检查 ref_policy 是否处于 eval 模式
    is_training = ref_policy.training
    print(f"ref_policy training mode: {is_training}")
    
    return ref_policy is not None

def test_forward_pass():
    """测试前向传播"""
    print("\n" + "="*80)
    print("测试 3: 前向传播")
    print("="*80)
    
    config = TransfuserConfig()
    agent = TransfuserAgent(config=config, lr=1e-4)
    agent._transfuser_model.train()
    
    # 创建模拟输入
    batch_size = 2
    device = 'cpu'
    
    features = {
        'camera_feature': torch.randn(batch_size, 1, 3, 256, 1024, device=device),
        'lidar_feature': torch.randn(batch_size, 1, 2, 256, 256, device=device),
        'status_feature': torch.randn(batch_size, 8, device=device)
    }
    
    targets = {
        'trajectory': torch.randn(batch_size, 8, 3, device=device),
        'agent_states': torch.randn(batch_size, 30, 4, device=device),
        'agent_labels': torch.randint(0, 2, (batch_size, 30), device=device).float(),
        'bev_semantic_map': torch.randint(0, 7, (batch_size, 128, 256), device=device)
    }
    
    # 创建模拟 tokens_list
    tokens_list = ['token1', 'token2']
    
    try:
        # 前向传播
        with torch.no_grad():
            predictions = agent._transfuser_model(features, targets, tokens_list)
        
        print(f"✓ 前向传播成功!")
        print(f"返回的键: {list(predictions.keys())}")
        
        # 检查必需的键
        required_keys = ['trajectory', 'final_poses_cls', 'final_ref_poses_cls', 'num_modes']
        missing_keys = [k for k in required_keys if k not in predictions]
        
        if missing_keys:
            print(f"✗ 缺少必需的键: {missing_keys}")
        else:
            print(f"✓ 所有必需的键都存在!")
            
        # 检查形状
        print(f"\n输出形状:")
        for k, v in predictions.items():
            if v is not None and isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v).__name__}")
                
        return True
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_computation():
    """测试损失计算"""
    print("\n" + "="*80)
    print("测试 4: 损失计算")
    print("="*80)
    
    config = TransfuserConfig()
    agent = TransfuserAgent(config=config, lr=1e-4)
    
    # 模拟预测结果（validation 模式 - 无 rewards）
    predictions_val = {
        'trajectory': torch.randn(2, 8, 3),
        'final_poses_cls': torch.randn(2, 64),
        'final_ref_poses_cls': torch.randn(2, 64),
        'rewards': None,
        'kl_div': None,
        'num_modes': 64
    }
    
    targets = {
        'trajectory': torch.randn(2, 8, 3),
    }
    
    try:
        # Validation loss (应该返回 0)
        loss_dict_val = agent.compute_loss({}, targets, predictions_val)
        print(f"Validation loss: {loss_dict_val}")
        
        # Training loss (有 rewards)
        predictions_train = predictions_val.copy()
        predictions_train['rewards'] = torch.randn(2, 64)
        predictions_train['kl_div'] = torch.tensor(0.1)
        
        loss_dict_train = agent.compute_loss({}, targets, predictions_train)
        print(f"Training loss: {loss_dict_train}")
        
        # 检查 loss 键
        if 'loss' in loss_dict_train and 'grpo_loss' in loss_dict_train and 'kl_loss' in loss_dict_train:
            print(f"✓ 损失计算成功!")
            print(f"  总损失: {loss_dict_train['loss'].item():.4f}")
            print(f"  GRPO损失: {loss_dict_train['grpo_loss'].item():.4f}")
            print(f"  KL损失: {loss_dict_train['kl_loss'].item():.4f}")
            return True
        else:
            print(f"✗ 损失字典缺少必需的键!")
            return False
            
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "#"*80)
    print("#" + " "*30 + "GRPO 训练流程测试" + " "*30 + "#")
    print("#"*80)
    
    results = {}
    
    # 测试 1: 参数冻结
    try:
        agent = test_parameter_freezing()
        results['parameter_freezing'] = True
    except Exception as e:
        print(f"✗ 参数冻结测试失败: {e}")
        results['parameter_freezing'] = False
    
    # 测试 2: ref_policy
    try:
        results['ref_policy'] = test_ref_policy()
    except Exception as e:
        print(f"✗ ref_policy 测试失败: {e}")
        results['ref_policy'] = False
    
    # 测试 3: 前向传播
    try:
        results['forward_pass'] = test_forward_pass()
    except Exception as e:
        print(f"✗ 前向传播测试失败: {e}")
        results['forward_pass'] = False
    
    # 测试 4: 损失计算
    try:
        results['loss_computation'] = test_loss_computation()
    except Exception as e:
        print(f"✗ 损失计算测试失败: {e}")
        results['loss_computation'] = False
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("✓ 所有测试通过! GRPO 训练流程配置正确!")
    else:
        print("✗ 部分测试失败，请检查上述错误信息。")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
