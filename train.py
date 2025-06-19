import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models import *
from tools import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
from tqdm import tqdm

def train_multimodal_model(model, train_loader, optimizer, criterion, device, epochs, generate=False):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in train_loader:
                # 获取数据
                images, labels = batch["image"].to(device)
                text = batch["text"].to(device)
                
                # 前向传播
                if generate:
                    img_logits, diag_logits, reports = model(images, text, generate=True)
                else:
                    img_logits, diag_logits = model(images, text)
                loss = criterion(img_logits, labels, diag_logits, text)
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        # 打印训练信息
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # 保存模型
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"multimodal_model_epoch_{epoch+1}.pth")

def inference(model, val_loader, optimizer, criterion, device,):
    model.eval()
    
    # 加载并预处理图像
    with tqdm(total=len(val_loader), desc="Inference") as pbar:
        for batch in val_loader:
            images, labels = batch["image"].to(device)
            text = batch["text"].to(device)
          
            with torch.no_grad():
                # 前向传播
                img_logits, diag_logits, reports = model(images, text, generate=True)
                # 计算损失
                loss = criterion(img_logits, labels, diag_logits, text)

            # 解码生成的报告
            tokenizer = model.text_encoder.tokenizer
            decoded_reports = [tokenizer.decode(report, skip_special_tokens=True) for report in reports]
            print("\n生成的医学报告示例:")
            for i, report in enumerate(decoded_reports):
                print(f"报告 {i+1}: {report}")
            pbar.set_postfix({"Loss": loss.item()})
            pbar.update(1)
    return loss

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalMultimodalSystem(num_classes=args.num_classes).to(device)
    
    # 准备数据
    train_dataset = MedicalMultimodalDataset(
        image_dir=args.image_dir, 
        text_dir=args.text_dir,
        augment=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 训练模型
    train_multimodal_model(model, train_loader, optimizer, criterion, device, epochs=args.epochs, generate=args.generate)

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Train and evaluate a multimodal medical model.")
    parse.add_argument('--image_dir',      type=str,   default="./datasets/images",      help='Directory containing images')
    parse.add_argument('--text_dir',       type=str,   default="./datasets/annotations", help='Directory containing text files')
    parse.add_argument('--epochs',         type=int,   default=10,       help='Number of training epochs')
    parse.add_argument('--batch_size',     type=int,   default=8,        help='Batch size for training')
    parse.add_argument('--lr',             type=float, default=1e-4,     help='Learning rate for optimizer')
    parse.add_argument('--num_classes',    type=int,   default=15,       help='Number of classes for segmentation and diagnosis')
    parse.add_argument('--generate',       type=bool,  default=False,    help='Whether to generate reports during inference')
    args = parse.parse_args()
    main(args)