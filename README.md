# PyPV-Eval: 光伏项目技术经济评价引擎

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Standard](https://img.shields.io/badge/Standard-NB%2FT%2011894--2025-red)](https://www.nea.gov.cn/)

**PyPV-Eval** 是一个轻量级、开源的光伏项目投资技经评价工具。它基于 Python 构建，严格遵循 **NB/T 11894-2025《光伏发电项目经济评价规范》**，旨在替代传统的封闭式 Windows 造价软件（如木联能）。

> **核心价值**：去黑盒化、云端运行、支持反向求解（Goal Seek）。

## 🚀 在线运行 (无需安装)

点击下方按钮，直接在 Google Colab 中运行本项目：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-14NFQ8rsNddUHDZTiRMN5L7ZdxWwRb9?usp=sharing)

## 🌟 核心功能

* **全生命周期计算**：自动生成 25 年现金流表（Cash Flow），涵盖建设期利息、流动资金注入与回收。
* **高精度税务模型**：
    * ✅ **增值税抵扣池**：自动处理进项税额抵扣逻辑。
    * ✅ **所得税优惠**：内置“三免三减半”政策算法。
* **智能反向求解 (Goal Seek)**：
    * 输入目标 IRR（如 8%），一键反推最大允许工程造价（Static Investment）。
* **合规性**：
    * 运维费率严格执行规范附录 A（阶梯式费率：10/18/28/32 元/kW）。

## 📊 案例验证 (Benchmark)

本项目已通过实际工程案例（琼海 100MW 集中式光伏）验证，结果与行业主流软件（木联能）对标：

| 指标项 | 目标值 (木联能) | PyPV-Eval 计算值 | 偏差 |
| :--- | :--- | :--- | :--- |
| **建设期利息** | 780.18 万元 | 780.18 万元 | ✅ 0.00 |
| **总投资** | 41080.18 万元 | 41080.18 万元 | ✅ 0.00 |
| **全投资 IRR (税前)** | 11.35% | 11.35% | ✅ 0.00% |

## 📜 许可证
本项目采用 MIT 开源许可证。
