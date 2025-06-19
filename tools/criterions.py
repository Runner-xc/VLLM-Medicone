import torch
import torch.nn as nn
import torch.nn.functional as F

class CE(nn.Module):
    def __init__(self, img_weight=1.0, diag_weight=1.0, report_weight=1.0):
        """
        多模态医学模型的多任务损失函数
        
        参数:
            img_weight: 图像预测损失的权重
            diag_weight: 诊断预测损失的权重
            report_weight: 报告生成损失的权重
        """
        super().__init__()
        self.img_weight = img_weight
        self.diag_weight = diag_weight
        self.report_weight = report_weight
        
        # 图像预测损失（假设是分类任务）
        self.img_criterion = nn.CrossEntropyLoss()
        
        # 诊断预测损失（假设是多标签分类）
        self.diag_criterion = nn.BCEWithLogitsLoss()
        
        # 报告生成损失（交叉熵）
        self.report_criterion = nn.CrossEntropyLoss(ignore_index=-100)  # 忽略填充token

    def forward(self, img_logits, labels, diag_logits, text, reports=None):
        """
        计算多任务损失
        
        参数:
            img_logits: 图像预测结果 [batch_size, num_classes]
            labels: 图像标签 [batch_size] 或 [batch_size, num_classes]
            diag_logits: 诊断预测结果 [batch_size, num_diseases]
            text: 目标文本的token IDs [batch_size, seq_len]
            reports: 生成的报告token IDs [batch_size, gen_len] (可选，用于推理阶段)
        
        返回:
            总损失
        """
        # 1. 图像预测损失
        # 假设labels是单标签分类（形状: [batch_size]）
        if labels.dim() == 1:
            img_loss = self.img_criterion(img_logits, labels)
        # 假设labels是多标签分类（形状: [batch_size, num_classes]）
        else:
            img_loss = self.diag_criterion(img_logits, labels)  # 复用BCE损失
        
        # 2. 诊断预测损失（假设是多标签分类）
        diag_loss = self.diag_criterion(diag_logits, text)  # 假设text是诊断标签
        
        # 3. 报告生成损失（仅在训练生成时计算）
        if reports is not None:
            # 获取目标文本的token IDs（不包括[CLS]）
            target_ids = text[:, 1:]  # [batch_size, seq_len-1]
            
            # 获取生成报告的token IDs（不包括起始token）
            gen_ids = reports[:, :-1]  # [batch_size, gen_len-1]
            
            # 展平为2D进行交叉熵计算
            gen_ids_flat = gen_ids.reshape(-1)  # [batch_size * (gen_len-1)]
            target_ids_flat = target_ids.reshape(-1)  # [batch_size * (seq_len-1)]
            
            # 计算报告生成损失
            report_loss = self.report_criterion(gen_ids_flat, target_ids_flat)
        else:
            report_loss = torch.tensor(0.0, device=img_logits.device)
        
        # 4. 加权组合总损失
        total_loss = (
            self.img_weight * img_loss +
            self.diag_weight * diag_loss +
            self.report_weight * report_loss
        ) 
        return total_loss