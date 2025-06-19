import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    PhrasalConstraint,  # 使用原始的PhrasalConstraint
    BioGptForCausalLM,  # 替换GPT2LMHeadModel
    BioGptTokenizer,
    GenerationConfig,
    LogitsProcessor  # 用于自定义约束
)
import torchvision.models as models
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# 自定义约束处理器(如果需要更灵活的控制)
class MedicalTermConstraint(LogitsProcessor):
    def __init__(self, tokenizer, medical_terms):
        self.tokenizer = tokenizer
        # 将医学术语转换为token IDs
        self.medical_term_ids = [
            tokenizer.encode(term, add_special_tokens=False)
            for term in medical_terms
        ]
        
    def __call__(self, input_ids, scores):
        # 这里可以实现自定义的约束逻辑
        # 例如：提高医学术语的概率,或确保生成的序列包含特定术语
        return scores  # 简化版：不做修改(实际应用中需要实现具体逻辑)

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = models.densenet121(pretrained=True)
        self.features = nn.Sequential(*list(self.densenet.children())[:-1])
        
        # 病灶注意力机制
        self.lesion_attn = nn.Sequential(
            nn.Conv2d(1024, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        feats = self.features(x)
        attn_map = self.lesion_attn(feats)
        return torch.sum(feats * attn_map, dim=[2,3])  # 5 次下采样 [B, 1024, H/32, W/32]]

class ClinicalTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.gate = nn.Linear(1024, 1)
        
        # 冻结部分层
        for layer in self.model.biogpt.layers[:6]:  
            for param in layer.parameters():
                param.requires_grad = False
            
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model.biogpt(  # 通过.biogpt访问模型核心
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        pooled = torch.mean(outputs.last_hidden_state, dim=1)  # [B, 1024]
        gate = torch.sigmoid(self.gate(pooled))
        return gate * pooled    # [B, 1024]
    
    def tokenize_text(self, text, max_length=512, return_tensors="pt"):
        """将文本转换为模型输入的token IDs和attention mask"""
        if isinstance(text, str):
            text = [text]  # 确保输入是列表
        
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors=return_tensors
        )
        return encoding["input_ids"], encoding["attention_mask"]

class CrossModalAttention(nn.Module):
    def __init__(self, img_dim=1024, text_dim=768):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, text_dim)
        self.text_proj = nn.Linear(img_dim, text_dim)
        self.cross_attn = nn.MultiheadAttention(text_dim, 8)
        
    def forward(self, img_feat, text_feat):
        projected_img = self.img_proj(img_feat)
        projected_text = self.text_proj(text_feat)
        fused, _ = self.cross_attn(
            query=projected_text.unsqueeze(1),
            key=projected_img.unsqueeze(1),
            value=projected_img.unsqueeze(1)
        )
        return fused.squeeze(1)

class MedicalMultimodalSystem(nn.Module):
    def __init__(self, num_classes=15, medical_terms=["pneumonia", "tumor"]):
        super().__init__()
        self.img_encoder = ImageEncoder()
        self.text_encoder = ClinicalTextEncoder()
        self.fusion = CrossModalAttention()
        
        # 多任务输出头（不变）
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
        
        self.diag_head = nn.Linear(768, num_classes)
        self.generation_config = GenerationConfig(
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        # 医学术语约束 - 修正输入格式（关键修改）
        tokenizer = self.text_encoder.tokenizer
        # 获取每个术语的token ID列表，形成二维列表 [[id1], [id2]]
        term_token_ids = [
            tokenizer.encode(term, add_special_tokens=False)
            for term in medical_terms
        ]
        print(f"医学术语约束的token IDs: {term_token_ids}")  # 输出: [[20673], [14601]]
        
        # 直接传入二维列表，创建一个PhrasalConstraint实例
        self.constraints = [
            PhrasalConstraint(token_ids)  # 每个术语的ID列表再包裹一层列表
            for token_ids in term_token_ids
        ]
        
        # 或者使用自定义约束处理器(更灵活)
        # self.medical_term_processor = MedicalTermConstraint(tokenizer, medical_terms)
        
    def forward(self, pixel_values, text=None, input_ids=None, attention_mask=None, generate=False):
        """
        前向传播函数
        
        参数:
            pixel_values: 图像张量 [batch_size, 3, 224, 224]
            text: 文本字符串或字符串列表(可选,与input_ids二选一)
            input_ids: 文本token IDs [batch_size, seq_len](可选,与text二选一)
            attention_mask: 注意力掩码 [batch_size, seq_len](当提供input_ids时必须提供)
            generate: 是否生成医学报告
            
        返回:
            seg_output: 分割结果 [batch_size, num_classes, 224, 224]
            diag_logits: 诊断分类 [batch_size, num_classes]
            reports: 生成的医学报告(仅当generate=True时)
        """
        # 图像编码
        img_feat = self.img_encoder(pixel_values)
        
        # 文本编码 - 支持两种输入方式
        if text is not None:
            # 如果提供了文本字符串,使用tokenizer转换为input_ids和attention_mask
            input_ids, attention_mask = self.text_encoder.tokenize_text(text)
            input_ids = input_ids.to(pixel_values.device)
            attention_mask = attention_mask.to(pixel_values.device)
        elif input_ids is None or attention_mask is None:
            raise ValueError("必须提供text或(input_ids和attention_mask)")
        
        text_feat = self.text_encoder(input_ids, attention_mask)
        
        # 多模态融合
        fused = self.fusion(img_feat, text_feat)
        
        # 分割任务
        img_feat_expanded = img_feat.unsqueeze(-1).unsqueeze(-1).expand(-1, 1024, 7, 7)
        seg_output = self.seg_head(img_feat_expanded)
        
        # 诊断分类
        diag_logits = self.diag_head(fused)
        
        # 生成医学报告(如果需要)
        if generate:
            reports = self.generate_report(img_feat, input_ids, attention_mask)
            return seg_output, diag_logits, reports
        
        return seg_output, diag_logits
        
    def generate_report(self, img_feat, input_ids, attention_mask):
        """使用图像特征和文本提示生成医学报告"""
        generated = self.text_encoder.model.generate(  # 直接使用BioGPT模型
            inputs=input_ids,  # 使用[CLS]作为起始token
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            inputs_embeds=self._fuse_features(img_feat, input_ids),
            constraints=self.constraints,
        )
        return generated
    
    def _fuse_features(self, img_feat, input_ids):
        """图像与文本特征融合（示例实现）"""
        # 将图像特征投影到文本空间
        img_feat = img_feat.unsqueeze(1).expand(-1, input_ids.size(1), -1)  # [B, seq_len, 1024]
        # 获取文本嵌入
        text_emb = self.text_encoder.model.biogpt.embed_tokens(input_ids)
        # 拼接特征（首帧为图像特征）
        return img_feat + text_emb  # [B, seq_len, 1024]

# 示例使用
if __name__ == "__main__":
    # 初始化模型
    model = MedicalMultimodalSystem(num_classes=15)
    model.eval()  # 设置为评估模式
    
    # 生成示例输入
    batch_size = 2
    
    # 图像输入
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    
    # 文本输入 - 使用字符串
    text = ["患者出现胸痛症状,伴有咳嗽和发热", "肺部CT显示右肺下叶有阴影"]
    
    # 前向传播
    with torch.no_grad():
        seg_output, diag_logits, reports = model(
            pixel_values=pixel_values,
            text=text,
            generate=True
        )
    
    # 打印结果
    print("分割输出形状:", seg_output.shape)  # [2, 15, 224, 224]
    print("诊断分类输出形状:", diag_logits.shape)  # [2, 15]
    print("生成报告形状:", reports.shape)  # [2, seq_len]
    
    # 解码生成的报告
    tokenizer = model.text_encoder.tokenizer
    decoded_reports = [tokenizer.decode(report, skip_special_tokens=True) for report in reports]
    print("\n生成的医学报告示例:")
    for i, report in enumerate(decoded_reports):
        print(f"报告 {i+1}: {report}")