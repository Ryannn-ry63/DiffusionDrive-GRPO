import torch
import argparse
import os

def convert_lightning_to_pth(ckpt_path, pth_path):
    """将Lightning .ckpt转换为普通 .pth格式"""
    
    print(f"🔍 开始转换: {ckpt_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print("✅ Checkpoint加载成功")
    print("📋 Checkpoint包含的键:", list(checkpoint.keys()))
    
    # 提取state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"📊 State dict包含 {len(state_dict)} 个参数")
        print("🔍 State dict前5个键:", list(state_dict.keys())[:5])
        
        # 移除Lightning添加的前缀
        new_state_dict = {}
        for key, value in state_dict.items():
            # 处理常见的Lightning前缀
            if key.startswith('model.'):
                new_key = key[6:]  # 移除 'model.'
            elif key.startswith('net.'):
                new_key = key[4:]  # 移除 'net.'
            elif key.startswith('_model.'):
                new_key = key[7:]  # 移除 '_model.'
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        print("🔄 转换后前5个键:", list(new_state_dict.keys())[:5])
        
        # 保存为.pth格式
        torch.save(new_state_dict, pth_path)
        print(f"✅ 转换成功！保存到: {pth_path}")
        print(f"📁 文件大小: {os.path.getsize(pth_path) / 1024 / 1024:.2f} MB")
        
    else:
        # 如果没有state_dict，可能是已经保存为state_dict格式
        torch.save(checkpoint, pth_path)
        print(f"✅ 直接保存为pth格式: {pth_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='输入的.ckpt文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出的.pth文件路径')
    args = parser.parse_args()
    
    convert_lightning_to_pth(args.input, args.output)