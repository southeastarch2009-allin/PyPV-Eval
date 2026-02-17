# PyPV-Eval: 光伏项目技术经济评价引擎 (改进版)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Standard](https://img.shields.io/badge/Standard-NB%2FT%2011894--2025-red)](https://www.nea.gov.cn/)

**PyPV-Eval** 是一个轻量级、开源的光伏项目投资技经评价工具。它基于 Python 构建，严格遵循 **NB/T 11894-2025《光伏发电项目经济评价规范》**，旨在替代传统的封闭式 Windows 造价软件（如木联能）。

> **核心价值**：去黑盒化、云端运行、支持反向求解（Goal Seek）。

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/southeastarch2009-allin/PyPV-Eval.git
cd PyPV-Eval

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

```bash
python3 main.py
```

## 📦 依赖项

- `pandas` >= 2.0.0
- `numpy` >= 1.24.0
- `numpy-financial` >= 1.1.0
- `scipy` >= 1.10.0

## 🌟 核心功能

### 1. 全生命周期计算
自动生成 25 年现金流表（Cash Flow），涵盖建设期利息、流动资金注入与回收。

### 2. 高精度税务模型
- ✅ **增值税抵扣池**：自动处理进项税额抵扣逻辑
- ✅ **所得税优惠**：内置"三免三减半"政策算法

### 3. 智能反向求解 (Goal Seek)
输入目标 IRR（如 8%），一键反推最大允许工程造价（Static Investment）

```python
from main import PVProject, goal_seek_investment

params = {
    'capacity_mw': 100.0,
    'static_invest': 40000.0,
    'hours': 1500,
    'price_tax_inc': 0.40
}

# 反向求解
max_invest = goal_seek_investment(8.0, params)
print(f"最大允许投资: {max_invest} 万元")
```

### 4. 合规性
运维费率严格执行规范附录 A（阶梯式费率：10/18/28/32 元/kW）

## 📊 API 参考

### PVProject 类

#### 初始化参数

| 参数名 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `capacity_mw` | float | 装机容量 (MW) | 必填 |
| `static_invest` | float | 静态投资 (万元) | 必填 |
| `hours` | float | 年利用小时数 (h) | 1000 |
| `loan_rate` | float | 长期贷款利率 | 0.049 |
| `capital_ratio` | float | 资本金比例 | 0.2 |
| `price_tax_inc` | float | 含税电价 (元/kWh) | 必填 |
| `deductible_tax` | float | 可抵扣进项税 (万元) | 自动计算 |

#### 方法

- `calculate_cash_flow()`: 计算现金流表
- `get_metrics()`: 获取核心指标

### 示例代码

```python
from main import PVProject

# 创建项目
params = {
    'capacity_mw': 100.0,
    'static_invest': 40000.0,
    'capital_ratio': 0.20,
    'loan_rate': 0.04876,
    'hours': 1500,
    'price_tax_inc': 0.40,
    'deductible_tax': 4000.0
}

project = PVProject(params)
project.calculate_cash_flow()
metrics = project.get_metrics()

print(f"总投资: {metrics['总投资']} 万元")
print(f"IRR(税前): {metrics['全投资IRR(税前)']}%")
```

## 📊 案例验证 (Benchmark)

本项目已通过实际工程案例（琼海 100MW 集中式光伏）验证：

| 指标项 | 目标值 (木联能) | PyPV-Eval 计算值 | 偏差 |
|:---|:---|:---|:---|
| **建设期利息** | 780.18 万元 | 780.18 万元 | ✅ 0.00 |
| **总投资** | 41080.18 万元 | 41080.18 万元 | ✅ 0.00 |
| **全投资 IRR (税前)** | 11.35% | 11.35% | ✅ 0.00% |

## 🔧 v1.0.2 改进内容

相比原版，改进版包含以下更新：

1. ✅ 修复了 bare except 子句
2. ✅ 添加了完整的类型注解
3. ✅ 添加了输入参数验证
4. ✅ 提取了魔法数字为常量
5. ✅ 添加了自定义异常类
6. ✅ 添加了日志系统
7. ✅ 改进了文档字符串
8. ✅ 添加了 `requirements.txt`

## 📜 许可证

本项目采用 MIT 开源许可证。
