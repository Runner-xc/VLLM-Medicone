"""
📁 文件结构生成器 (metadata_builder.py)
功能:根据现有医学影像数据生成规范化的CSV索引、JSON/TXT模板文件
"""
import os
import json
import csv
from pathlib import Path

def validate_structure(root_dir):
    """验证并生成标准目录结构"""
    required_dirs = ['images', 'masks', 'annotations']
    for d in required_dirs:
        dir_path = Path(root_dir) / d
        dir_path.mkdir(exist_ok=True)
        print(f"✅ 已创建/验证目录：{dir_path}")

def generate_metadata(root_dir):
    """主生成函数"""
    root = Path(root_dir)
    csv_data = []
    
    # 遍历所有影像文件
    for img_file in (root / 'images').glob('*.*'):
        if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        # 生成路径信息 
        case_id = img_file.stem.split('_')[0]  # 假设文件名格式：case_001.jpg
        base_info = {
            'image_id': case_id,
            'image_path': str(img_file.relative_to(root)),
            'mask_path': str((root / 'masks' / f'{case_id}_mask.png').relative_to(root)) 
                        if (root / 'masks' / f'{case_id}_mask.png').exists() else "",
            'annotation_path': str((root / 'annotations' / f'{case_id}.json').relative_to(root))
        }

        # 生成JSON模板 
        json_path = root / base_info['annotation_path']
        if not json_path.exists():
            json_template = {
                "image_id": case_id,
                "image_path": base_info['image_path'],
                "segmentation_info": {
                    "lesion_area": "",
                    "size_mm": "",
                    "location": ""
                },
                "pathology_analysis": "",
                "diagnosis_suggestion": [],
                "medical_tags": []
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_template, f, indent=2, ensure_ascii=False)
            print(f"📄 生成JSON模板：{json_path}")

        # 生成TXT模板 
        txt_path = root / 'annotations' / f'{case_id}.txt'
        if not txt_path.exists():
            txt_template = (
                "pathology_analysis: \n"
                "diagnosis_suggestion: []"
            )
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(txt_template)
            print(f"📝 生成TXT模板：{txt_path}")

        csv_data.append(base_info)

    # 生成CSV索引 
    csv_path = root / 'dataset.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'image_path', 'mask_path', 'annotation_path'])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\n🎉 元数据生成完成！共处理 {len(csv_data)} 个病例")
    print(f"CSV文件路径：{csv_path.resolve()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="医学元数据生成器")
    parser.add_argument("-d", "--directory", type=str, default="/mnt/e/VScode/WS-Hub/Linux-VLLM-Med/VLLM-Medicone/datasets", 
                       help="数据集根目录路径")
    args = parser.parse_args()
    
    validate_structure(args.directory)
    generate_metadata(args.directory)
