"""
Fruit-306 Dataset Loader
从4.py提取的数据集加载模块，支持从CSV加载类别映射和水果描述
"""

import os
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import logging

# 配置日志
logger = logging.getLogger(__name__)

class Fruit306Dataset(Dataset):
    """
    Fruit-306数据集加载器
    支持从train/val/test目录结构加载图像
    """
    def __init__(self, root_dir, split='train', transform=None, class_mapping_path=None):
        """
        Args:
            root_dir: 数据集根目录，包含 train/val/test 子文件夹
            split: 'train', 'val', 或 'test'
            transform: 图像变换
            class_mapping_path: 类别映射CSV文件路径（可选）
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.split_dir = self.root_dir / split
        
        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")
        
        # 获取类别列表（按文件夹名称排序）
        self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 加载所有图像路径和标签
        self.images = []
        self.labels = []
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for class_name in self.classes:
            class_path = self.split_dir / class_name
            label = self.class_to_idx[class_name]
            
            for img_file in class_path.iterdir():
                if img_file.suffix.lower() in valid_extensions:
                    self.images.append(str(img_file))
                    self.labels.append(label)
        
        logger.info(f"[Dataset] Loaded {len(self.images)} images from {split} split, {len(self.classes)} classes")
        
        # 如果提供了class_mapping_path，加载水果描述
        self.fruit_descriptions = {}
        if class_mapping_path and os.path.exists(class_mapping_path):
            self._load_fruit_descriptions(class_mapping_path)
    
    def _load_fruit_descriptions(self, mapping_path):
        """从CSV加载水果描述"""
        try:
            df = pd.read_csv(mapping_path)
            
            # 自动检测class_name列
            if 'class_name' in df.columns:
                col_name = 'class_name'
            elif 'class' in df.columns:
                col_name = 'class'
            elif 'label' in df.columns:
                col_name = 'label'
            else:
                col_name = df.columns[1] if 'class_index' in df.columns else df.columns[0]
            
            # 加载description列
            default_desc = "未知特征，请基于图像分析颜色、形状、纹理、斑点等视觉细节"
            if 'description' in df.columns:
                for _, row in df.iterrows():
                    cls_name = row[col_name]
                    desc = row['description'] if pd.notna(row['description']) else default_desc
                    self.fruit_descriptions[cls_name] = desc
                logger.info(f"[Dataset] Loaded {len(self.fruit_descriptions)} fruit descriptions from CSV")
            else:
                logger.warning("[Dataset] No 'description' column found in CSV")
        except Exception as e:
            logger.error(f"[Dataset] Failed to load descriptions: {e}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            # 返回一个占位符，或者重新尝试加载
            raise
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_name(self, idx):
        """通过索引获取类别名称"""
        return self.classes[idx]
    
    def get_description(self, class_name):
        """获取水果描述"""
        return self.fruit_descriptions.get(class_name, "")


def load_test_dataset(test_dir, class_mapping_path=None):
    """
    加载测试集图像路径和标签（用于评估）
    
    Args:
        test_dir: 测试集目录路径
        class_mapping_path: 类别映射CSV路径
    
    Returns:
        samples: 列表，每个元素为 (image_path, label)
        class_names: 类别名称列表
        class_to_idx: 类别到索引的映射
    """
    samples = []
    test_path = Path(test_dir)
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # 加载类别映射
    class_names = None
    class_to_idx = None
    
    if class_mapping_path and os.path.exists(class_mapping_path):
        class_names, class_to_idx, _ = verify_class_mapping(class_mapping_path)
    else:
        # 从目录结构推断类别
        class_names = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
        class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    for cls_dir in test_path.iterdir():
        if not cls_dir.is_dir():
            continue
        
        cls_name = cls_dir.name
        if cls_name not in class_to_idx:
            continue
        
        label = class_to_idx[cls_name]
        for img_file in cls_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                samples.append((str(img_file), label))
    
    logger.info(f"[Dataset] Loaded {len(samples)} test samples")
    return samples, class_names, class_to_idx


def verify_class_mapping(path):
    """
    验证并加载类别映射文件，包括description列
    
    Args:
        path: CSV文件路径
    
    Returns:
        class_names: 类别名称列表
        class_to_idx: 类别到索引的映射
        num_classes: 类别数量
        descriptions: 水果描述字典
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Class mapping file not found: {path}")
    
    df = pd.read_csv(path)
    
    # 自动检测class_name列
    if 'class_name' in df.columns:
        col_name = 'class_name'
    elif 'class' in df.columns:
        col_name = 'class'
    elif 'label' in df.columns:
        col_name = 'label'
    else:
        col_name = df.columns[1] if 'class_index' in df.columns else df.columns[0]
    
    class_names = df[col_name].astype(str).tolist()
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    logger.info(f"[Dataset] Loaded class mapping: {len(class_names)} classes")
    
    # 加载description到字典
    descriptions = {}
    default_desc = "未知特征，请基于图像分析颜色、形状、纹理、斑点等视觉细节"
    
    if 'description' in df.columns:
        for _, row in df.iterrows():
            cls_name = row[col_name]
            desc = row['description'] if pd.notna(row['description']) else default_desc
            descriptions[cls_name] = desc
        logger.info(f"[Dataset] Loaded {len(descriptions)} fruit descriptions")
    else:
        logger.warning("[Dataset] No 'description' column in CSV, using defaults")
    
    return class_names, class_to_idx, len(class_names), descriptions


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例用法
    dataset = Fruit306Dataset(
        root_dir="./Fruit-306",
        split='train',
        class_mapping_path="./fruit_name_mapping.txt"
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    print(f"First 5 classes: {dataset.classes[:5]}")