# 版本管理规范

## 核心原则：版本隔离

每个版本使用独立目录，互不干扰，便于对比和回退。

## 目录结构

```
improved_plastic_net/
├── README.md                    # 项目总览
├── VERSION_MANAGEMENT_GUIDE.md  # 本规范
├── requirements.txt             # 共享依赖
├── data/                        # 共享数据集
│
├── v1.0.2/                      # 版本1.0.2（当前稳定版）
│   ├── README.md
│   ├── config.py
│   ├── models/
│   ├── training/
│   ├── experiments/
│   ├── utils/
│   └── results/
│
├── v1.0.3/                      # 版本1.0.3（新版本）
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── config.py
│   └── ...
│
└── archive/                     # 归档旧版本
    └── v1.0.0/
```

## 新版本创建流程

### 1. 确定版本号
- Bug修复：v1.0.2 → v1.0.3
- 新功能：v1.0.2 → v1.1.0
- 重大变更：v1.0.2 → v2.0.0

### 2. 创建新版本目录
```bash
# 从当前版本复制
cp -r v1.0.2 v1.0.3

# 或选择性复制
mkdir -p v1.0.3/{models,training,experiments,utils,results}
cp v1.0.2/config.py v1.0.3/
cp v1.0.2/run_with_config.py v1.0.3/
```

### 3. 更新版本信息
在 config.py 中添加：
```python
__version__ = "1.0.3"
__version_date__ = "2026-02-04"
```

### 4. 创建 CHANGELOG.md
记录本版本的所有变更。

### 5. 在新版本目录中开发
所有修改只在新版本目录中进行。

## 运行特定版本

```bash
# 运行 v1.0.2
cd v1.0.2 && python run_with_config.py

# 运行 v1.0.3
cd v1.0.3 && python run_with_config.py
```

## 版本对比

```bash
# 对比代码
diff -r v1.0.2/models v1.0.3/models

# 对比结果
diff v1.0.2/results/performance.json v1.0.3/results/performance.json
```

## 注意事项

✅ **应该做的**：
- 每个版本独立目录
- 保留完整可运行代码
- 记录详细变更日志
- 共享数据集（避免重复）

❌ **不应该做的**：
- 直接修改旧版本
- 混合版本代码
- 删除工作版本
- 跨版本引用代码

---
创建日期：2026-02-03
