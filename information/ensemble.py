"""
FruitEnsemble: 四模型集成系统
从4.py提取的集成模型核心代码
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import logging
from collections import OrderedDict
import numpy as np
import random

# 配置日志
logger = logging.getLogger(__name__)

# 随机种子固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 设备选择
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Vision Transformer 组件定义 ====================

class PatchEmbedding(nn.Module):
    """将图像分割为 Patch 并嵌入为向量序列"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.projection = nn.Linear(self.patch_dim, embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(p=0.1)
        self._init_pos_embed()

    def _init_pos_embed(self):
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        # Unfold 操作将图像切分为 patch
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, 3, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(batch_size, self.num_patches, -1)
        x = self.projection(x)
        # 添加 Class Token
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_tokens, x], dim=1)
        # 添加位置编码
        x = x + self.position_embedding
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_length, _ = x.shape
        qkv = self.qkv(x).reshape(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        context = (attention_probs @ v).transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.embed_dim)
        return self.projection(context)


class MLP(nn.Module):
    """Transformer 中的前馈神经网络"""
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.1):
        super(MLP, self).__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder 单层"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-LN 架构
        attn_out = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        mlp_out = self.mlp(self.norm2(x))
        x = x + self.dropout(mlp_out)
        return x


class VisionTransformer(nn.Module):
    """完整的 Vision Transformer 模型"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=306, 
                 embed_dim=768, num_heads=12, num_layers=12, mlp_ratio=4.0, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout_rate)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.norm(x)
        # 取 Class Token 的输出
        cls_token = x[:, 0]
        return self.head(self.dropout(cls_token))


class VisionTransformerSmall(VisionTransformer):
    """小型 ViT 变体"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=306,
                 embed_dim=384, num_heads=6, num_layers=8, mlp_ratio=4.0, dropout_rate=0.1):
        super(VisionTransformerSmall, self).__init__(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, 
            num_classes=num_classes, embed_dim=embed_dim, num_heads=num_heads, 
            num_layers=num_layers, mlp_ratio=mlp_ratio, dropout_rate=dropout_rate
        )


# ==================== 模型加载器 ====================

def load_resnet50(path, num_classes):
    """加载 ResNet50 模型"""
    model = models.resnet50(weights=None)
    # 替换全连接层以匹配类别数
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        # 处理 DataParallel 前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"[LOAD] ResNet50 weights loaded: {path}")
    else:
        logger.warning(f"[WARN] ResNet50 weights not found: {path}")
    
    return model


def load_densenet201(path, num_classes):
    """加载 DenseNet201 模型"""
    model = models.densenet201(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Dropout(0.3),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, num_classes)
    )
    
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"[LOAD] DenseNet201 weights loaded: {path}")
    else:
        logger.warning(f"[WARN] DenseNet201 weights not found: {path}")
    
    return model


def load_vit(path, num_classes):
    """加载自定义 Vision Transformer 模型"""
    if not os.path.exists(path):
        logger.warning(f"[WARN] ViT weights not found: {path}")
        return VisionTransformer(num_classes=num_classes)
    
    checkpoint = torch.load(path, map_location='cpu')
    
    # 尝试从 checkpoint 中恢复配置
    config = checkpoint.get('config', checkpoint.get('args', {}))
    if hasattr(config, '__dict__'):
        config = vars(config)
    
    # 默认配置
    params = {
        'img_size': 224,
        'embed_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'dropout_rate': 0.1
    }
    
    # 更新配置
    if isinstance(config, dict):
        for k in params:
            if k in config:
                params[k] = config[k]
    
    # 根据 embed_dim 选择模型大小
    if params['embed_dim'] <= 384:
        model = VisionTransformerSmall(num_classes=num_classes, **params)
    else:
        model = VisionTransformer(num_classes=num_classes, **params)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        # 忽略分类头形状不匹配的键
        if 'head.weight' in name or 'head.bias' in name:
            if v.shape[0] != num_classes:
                continue
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    logger.info(f"[LOAD] ViT weights loaded: {path}")
    return model


def load_efficientnetb7(path, num_classes):
    """加载 EfficientNetB7 模型"""
    model = models.efficientnet_b7(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(in_features, 2048),
        nn.BatchNorm1d(2048),
        nn.SiLU(inplace=True),
        nn.Dropout(0.15, inplace=True),
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.SiLU(inplace=True),
        nn.Dropout(0.15, inplace=True),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.SiLU(inplace=True),
        nn.Dropout(0.075, inplace=True),
        nn.Linear(512, num_classes)
    )
    
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"[LOAD] EfficientNetB7 weights loaded: {path}")
    else:
        logger.warning(f"[WARN] EfficientNetB7 weights not found: {path}")
    
    return model


# ==================== 工具函数 ====================

def apply_temperature_scaling(logits, temperature=1.0):
    """应用温度缩放以校准概率分布"""
    if temperature <= 0:
        temperature = 1.0
    return logits / temperature


def get_base_transforms(model_name, img_size=224):
    """获取各模型对应的基础变换"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if model_name == "EfficientNetB7":
        return transforms.Compose([
            transforms.Resize((600, 600), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    elif model_name == "ViT":
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])
    else:  # ResNet, DenseNet
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])


def generate_tta_views(img, model_name, img_size=224):
    """生成测试时增强 (TTA) 视图，旋转时使用白色填充"""
    target_size = 600 if model_name == "EfficientNetB7" else (img_size if model_name == "ViT" else 224)
    resize_trans = transforms.Resize((target_size, target_size), interpolation=Image.BICUBIC)
    # 使用白色填充旋转产生的空缺
    fill_color = (255, 255, 255)
    views = []
    
    # 1. 原图
    views.append(resize_trans(img))
    # 2. 水平翻转
    views.append(resize_trans(img.transpose(Image.FLIP_LEFT_RIGHT)))
    # 3. 旋转 +15 度
    views.append(resize_trans(img.rotate(15, resample=Image.BICUBIC, fillcolor=fill_color)))
    # 4. 旋转 -15 度
    views.append(resize_trans(img.rotate(-15, resample=Image.BICUBIC, fillcolor=fill_color)))
    # 5. 中心裁剪 (Zoom In)
    try:
        scale_factor = 1.15
        new_size = int(target_size * scale_factor)
        resized = img.resize((new_size, new_size), resample=Image.BICUBIC)
        cropped = transforms.CenterCrop(size=target_size)(resized)
        views.append(cropped)
    except Exception:
        views.append(resize_trans(img))
    
    return views


# ==================== 集成模型主类 ====================

class FruitEnsemble:
    """
    FruitEnsemble: 四模型集成系统主类
    """
    def __init__(self, model_paths, num_classes, weights=None, device=DEVICE):
        """
        Args:
            model_paths: 字典，包含四个模型的路径
            num_classes: 类别数量
            weights: 各模型权重，如果为None则使用基于准确率的科学权重
            device: 运行设备
        """
        self.device = device
        self.num_classes = num_classes
        self.models_dict = {}
        
        # 基于准确率的科学权重 (DenseNet:0.685, ResNet:0.650, EffNet:0.628, ViT:0.597)
        self.base_weights = {
            "DenseNet201": 0.32,
            "ResNet50": 0.30,
            "EfficientNetB7": 0.23,
            "ViT": 0.15
        }
        
        # 加载模型
        self._load_models(model_paths)
        
        # 设置权重
        if weights is None:
            # 只保留已加载模型的权重并归一化
            available_weights = {k: self.base_weights.get(k, 0.25) for k in self.models_dict.keys()}
            total_weight = sum(available_weights.values())
            self.weights = {k: v / total_weight for k, v in available_weights.items()}
        else:
            self.weights = weights
        
        logger.info(f"[Ensemble] Final weights: {self.weights}")
    
    def _load_models(self, model_paths):
        """加载所有模型"""
        model_loaders = [
            ("ResNet50", model_paths.get("ResNet50"), load_resnet50, 224),
            ("DenseNet201", model_paths.get("DenseNet201"), load_densenet201, 224),
            ("ViT", model_paths.get("ViT"), load_vit, 224),
            ("EfficientNetB7", model_paths.get("EfficientNetB7"), load_efficientnetb7, 600)
        ]
        
        for name, path, loader_func, img_size in model_loaders:
            if path and os.path.exists(path):
                try:
                    model = loader_func(path, self.num_classes)
                    model = model.to(self.device)
                    model.eval()
                    self.models_dict[name] = {
                        "model": model,
                        "img_size": img_size
                    }
                    logger.info(f"[Ensemble] Model {name} loaded successfully")
                except Exception as e:
                    logger.error(f"[Ensemble] Failed to load {name}: {e}")
            else:
                logger.warning(f"[Ensemble] Model {name} weights not found, skipping")
    
    def predict(self, image, temperature=1.0, use_tta=True, return_topk=3):
        """
        单张图片预测
        
        Args:
            image: PIL Image 对象
            temperature: 温度缩放系数
            use_tta: 是否使用测试时增强
            return_topk: 返回前k个预测
        
        Returns:
            probs: 所有类别的概率
            topk_results: 前k个 (类别名, 概率) 列表
            model_details: 各模型预测详情
        """
        if not self.models_dict:
            raise RuntimeError("No models loaded")
        
        ensemble_probs_list = []
        model_details = {}
        
        # 标准化变换 (用于 TTA 内部)
        to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 多模型推理
        for name, cfg in self.models_dict.items():
            model = cfg["model"]
            img_size = cfg.get("img_size", 224)
            
            if use_tta:
                views = generate_tta_views(image, name, img_size)
            else:
                views = [image]
            
            view_probs = []
            for v_img in views:
                try:
                    input_tensor = to_tensor_norm(v_img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        logits = model(input_tensor)[0]
                    
                    scaled_logits = apply_temperature_scaling(logits, temperature)
                    probs = F.softmax(scaled_logits, dim=0)
                    view_probs.append(probs.cpu())
                except Exception as e:
                    logger.warning(f"[WARN] Model {name} TTA view failed: {e}")
                    continue
            
            if view_probs:
                avg_prob = torch.stack(view_probs).mean(dim=0)
                ensemble_probs_list.append(avg_prob)
                
                top1_idx = avg_prob.argmax().item()
                top1_prob = avg_prob[top1_idx].item()
                model_details[name] = {
                    "top1_class_idx": top1_idx,
                    "top1_prob": round(top1_prob, 4)
                }
        
        if not ensemble_probs_list:
            raise RuntimeError("All models failed")
        
        # 加权集成
        final_probs = torch.zeros(self.num_classes)
        weight_sum = 0.0
        
        for i, name in enumerate(self.models_dict.keys()):
            if name in model_details:
                w = self.weights.get(name, 0.25)
                final_probs += w * ensemble_probs_list[i]
                weight_sum += w
        
        if weight_sum > 0:
            final_probs /= weight_sum
        
        # 获取 Top-K
        topk_vals, topk_indices = torch.topk(final_probs, k=min(return_topk, self.num_classes))
        topk_results = [
            (idx.item(), round(val.item(), 4))
            for val, idx in zip(topk_vals, topk_indices)
        ]
        
        return final_probs, topk_results, model_details
    
    def get_model_count(self):
        """返回成功加载的模型数量"""
        return len(self.models_dict)


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 示例：加载模型
    model_paths = {
        "ResNet50": "./pretrained/resnet50_fruit306.pth",
        "DenseNet201": "./pretrained/densenet201_fruit306.pth",
        "ViT": "./pretrained/vit_fruit306.pth",
        "EfficientNetB7": "./pretrained/efficientnetb7_fruit306.pth"
    }
    
    ensemble = FruitEnsemble(model_paths, num_classes=306)
    
    # 示例预测
    if ensemble.get_model_count() > 0:
        img = Image.open("test_image.jpg").convert('RGB')
        probs, topk, details = ensemble.predict(img, use_tta=True)
        print(f"Top-3 predictions: {topk}")