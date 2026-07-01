

import numpy as np
import cv2
import os
import torch
from math import ceil
import torch.nn.functional as F

class SpherePatcher:
    @staticmethod
    def generate_patches(theta_range=(-90, 90), delta_theta=20, min_phi_div=4):
        """严格分块生成器（确保完全覆盖球面）"""
        patches = []
        theta_start = np.deg2rad(90 - theta_range[1])
        theta_end = np.deg2rad(90 - theta_range[0])
        
        theta_bins = []
        current = theta_start
        while True:
            theta_bins.append(current)
            next_step = np.deg2rad(delta_theta)
            if current + next_step >= theta_end:
                theta_bins.append(theta_end)
                break
            current += next_step
        
        for i in range(len(theta_bins)-1):
            theta_center = (theta_bins[i] + theta_bins[i+1])/2
            circumference = 2 * np.pi * np.sin(theta_center)
            num_phi = max(int(ceil(circumference / np.deg2rad(delta_theta))), min_phi_div)
            num_phi = num_phi if num_phi % 2 == 0 else num_phi + 1
            
            phi_step = 2 * np.pi / num_phi
            for j in range(num_phi):
                phi_min = j * phi_step
                phi_max = (j+1) * phi_step
                patches.append({
                    'theta_min': theta_bins[i],
                    'theta_max': theta_bins[i+1],
                    'phi_min': phi_min,
                    'phi_max': phi_max
                })
        return patches

def erp_to_patches(erp_tensor, patches):
    """
    将ERP张量转换为分块张量
    输入: [B, C, H, W]
    输出: [B, num_patches, C, 16, 16]
    """
    B, C, H, W = erp_tensor.shape
    device = erp_tensor.device
    patch_list = []
    meta_list = []
    for patch in patches:
        # 计算ERP坐标范围
        theta_min = patch['theta_min']
        theta_max = patch['theta_max']
        phi_min = patch['phi_min']
        phi_max = patch['phi_max']
        y_min = int(round(theta_min / np.pi * (H-1)))
        y_max = int(round(theta_max / np.pi * (H-1)))
        y_min, y_max = sorted([y_min, y_max])
        y_max = min(H-1, y_max)
        x_min = int(round(phi_min / (2*np.pi) * (W-1)))
        x_max = int(round(phi_max / (2*np.pi) * (W-1)))
        x_max = x_max % W
        # 提取原始分块
        if x_min <= x_max:
            patch_tensor = erp_tensor[:, :, y_min:y_max+1, x_min:x_max+1]
            original_w = x_max - x_min + 1
        else:
            left_part = erp_tensor[:, :, y_min:y_max+1, x_min:]
            right_part = erp_tensor[:, :, y_min:y_max+1, :x_max+1]
            patch_tensor = torch.cat([left_part, right_part], dim=3)
            original_w = (W - x_min) + (x_max + 1)
        original_h = y_max - y_min + 1
        # 展平为16x16
        flattened = F.interpolate(patch_tensor, size=(16, 16), mode='bilinear', align_corners=False)
        patch_list.append(flattened)
        # 保存元数据
        meta_list.append({
            'original_shape': (original_h, original_w),
            'position': (y_min, y_max, x_min, x_max)
        })
    return torch.stack(patch_list, dim=1), meta_list

def reconstruct_from_patches(patches_tensor, meta_list, original_shape):
    """
    从分块张量重建ERP图像
    输入: [B, num_patches, C, 16, 16]
    输出: [B, C, H, W]
    """
    B, N, C, _, _ = patches_tensor.shape
    H, W = original_shape
    device = patches_tensor.device
    erp_accum = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)
    erp_count = torch.zeros((B, 1, H, W), dtype=torch.float32, device=device)
    for i in range(N):
        patch = patches_tensor[:, i]
        meta = meta_list[i]
        oh, ow = meta['original_shape']
        y_min, y_max, x_min, x_max = meta['position']
        # 恢复原始尺寸
        resized = F.interpolate(patch, size=(oh, ow), mode='bilinear', align_corners=False)
        # 累加到对应位置
        if x_min <= x_max:
            erp_accum[:, :, y_min:y_max+1, x_min:x_max+1] += resized
            erp_count[:, :, y_min:y_max+1, x_min:x_max+1] += 1
        else:
            left_w = W - x_min
            right_w = x_max + 1
            left_part = resized[:, :, :, :left_w]
            right_part = resized[:, :, :, left_w:left_w+right_w]
            erp_accum[:, :, y_min:y_max+1, x_min:] += left_part
            erp_count[:, :, y_min:y_max+1, x_min:] += 1
            erp_accum[:, :, y_min:y_max+1, :right_w] += right_part
            erp_count[:, :, y_min:y_max+1, :right_w] += 1
    # 处理未覆盖区域
    erp_count[erp_count == 0] = 1
    # reconstructed = (erp_accum / erp_count).to(torch.uint8)
    reconstructed = erp_accum / erp_count
    return reconstructed

def tensor_to_canvas(erp_tensor, patches, canvas_path):
    """生成中间对齐的画布（PyTorch张量版本）"""
    B, C, H, W = erp_tensor.shape
    erp_np = erp_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    # 组织纬度分带
    theta_groups = []
    current_theta = None
    for patch in patches:
        if (patch['theta_min'], patch['theta_max']) != current_theta:
            theta_groups.append([])
            current_theta = (patch['theta_min'], patch['theta_max'])
        theta_groups[-1].append(patch)
    
    # 计算画布尺寸
    max_cols = max(len(g) for g in theta_groups)
    canvas_height = len(theta_groups) * 256
    canvas_width = max_cols * 256
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 按纬度分带居中排列
    for row_idx, group in enumerate(theta_groups):
        row_width = len(group) * 256
        start_x = (canvas_width - row_width) // 2
        
        for col_idx, patch in enumerate(group):
            # 提取原始分块
            y_min = int(round(patch['theta_min'] / np.pi * (H-1)))
            y_max = int(round(patch['theta_max'] / np.pi * (H-1)))
            y_min, y_max = sorted([y_min, y_max])
            y_max = min(H-1, y_max)
            
            x_min = int(round(patch['phi_min'] / (2*np.pi) * (W-1)))
            x_max = int(round(patch['phi_max'] / (2*np.pi) * (W-1)))
            x_max = x_max % W
            
            if x_min <= x_max:
                patch_np = erp_np[y_min:y_max+1, x_min:x_max+1]
            else:
                left = erp_np[y_min:y_max+1, x_min:]
                right = erp_np[y_min:y_max+1, :x_max+1]
                patch_np = np.hstack((left, right))
            
            # 缩放并填充到画布
            resized = cv2.resize(patch_np, (256, 256), interpolation=cv2.INTER_LINEAR)
            canvas[row_idx*256:(row_idx+1)*256, 
                   start_x+col_idx*256:start_x+(col_idx+1)*256] = resized
    
    cv2.imwrite(canvas_path, canvas)
    print(f"Canvas saved to: {canvas_path}")

if __name__ == "__main__":
    # 配置路径
    ERP_PATH = "/home/aistudio/image/0385.jpg"
    RECONSTRUCT_PATH = "/home/aistudio/reconstruct/reconstructed_pytorch2.jpg"
    CANVAS_PATH = "/home/aistudio/reconstruct/latlon_canvas_pytorch2.jpg"
    
    # 读取图像并转换为张量
    erp_img = cv2.imread(ERP_PATH)
    erp_tensor = torch.from_numpy(erp_img).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
    erp_tensor = erp_tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(erp_tensor.shape)
    
    # 生成分块参数
    patches = SpherePatcher.generate_patches(delta_theta=20, min_phi_div=4)
    
    # 转换为分块张量
    patches_tensor, meta_list = erp_to_patches(erp_tensor, patches)
    print(f"Generated patches tensor shape: {patches_tensor.shape}")
    
    # 重建ERP图像
    reconstructed = reconstruct_from_patches(patches_tensor, meta_list, erp_tensor.shape[2:])
    
    # 保存重建结果
    reconstructed_np = reconstructed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(RECONSTRUCT_PATH, reconstructed_np)
    print(f"Reconstruction saved to: {RECONSTRUCT_PATH}")
    
    # 生成并保存画布
    tensor_to_canvas(erp_tensor, patches, CANVAS_PATH)