### AI辅助诊疗系统

##### 数据集结构
medical_dataset/
├── images/              # 病理图片文件夹
│   ├── case_001.jpg     # 患者1的病理切片图像
│   ├── case_002.jpg     # 患者2的图像
│   └── ...              
├── masks/               # 分割掩码（可选，用于训练分割模型）
│   ├── case_001_mask.png # 对应图像的病灶分割标注
│   └── ...              
├── annotations/         # 文本标注文件
│   ├── case_001.json     # 包含图像描述、病理分析、诊断建议的JSON
│   ├── case_001.txt      # 也可使用TXT格式，每行对应一个标注字段
│   └── ...              
└── dataset.csv          # 数据集索引表（关键！便于模型读取）

`json文件`格式要求：
```text
{
  "image_id": "case_001",              # 图像唯一标识
  "image_path": "images/case_001.jpg", # 图像路径
  "segmentation_info": {              # 分割相关描述（可关联掩码）
    "lesion_area": "左上区域可见异常细胞团",
    "size_mm": "5.2×3.8",
    "location": "胃部黏膜层"
  },
  "pathology_analysis": "细胞形态不规则，核质比增大，符合腺癌特征",  # 病理分析文本
  "diagnosis_suggestion": [            # 诊断建议（列表形式，便于NLP处理）
    "建议进一步免疫组化检测",
    "考虑手术切除可能性",
    "定期复查胃镜"
  ],
  "medical_tags": ["腺癌", "胃部病变", "细胞异常"]  # 医学标签（可选，用于分类任务）
}
```

`txt`格式要求：
```text
"pathology_analysis": "在{部位}检测到{细胞形态}，{特征1}，{特征2}，{诊断方向}。",
"diagnosis_suggestion": ["建议{检查手段}进一步确认", "{治疗方向}评估", "定期{复查项目}监测"]
```

`csv文件`格式要求：
```text
image_id,image_path,mask_path,annotation_path
case_001,images/case_001.jpg,masks/case_001.png,annotations/case_001.json
case_002,images/case_002.jpg,masks/case_002.png,annotations/case_002.json
```
