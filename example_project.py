#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyPV-Eval 项目计算脚本示例

本脚本展示如何使用 PyPV-Eval 进行完整的光伏项目技经评价，
包括生成各类财务报表和敏感性分析。

使用方法:
    python example_project.py
"""

import sys
from main import PVProject, goal_seek_investment, sensitivity_analysis


# ==============================================================================
# 项目参数配置 - 请根据你的项目修改以下参数
# ==============================================================================

def get_project_params():
    """
    配置项目参数

    根据你的项目情况修改以下参数
    """
    return {
        # ========== 基础参数 ==========
        'capacity_mw': 10.0,              # 装机容量 (MW) - 👈 请修改
        'static_invest': 3500.0,          # 静态投资 (万元) - 👈 请修改

        # ========== 选择收益模式 ==========
        'mode': 'full_grid',              # 'full_grid'=全额上网, 'self_consumption'=自发自用

        # ========== 全额上网模式参数 ==========
        'price_tax_inc': 0.38,            # 上网电价 (元/kWh) - 👈 请修改
        'hours': 1300,                    # 年利用小时数 (h) - 👈 请修改

        # ========== 自发自用模式参数（如果mode='self_consumption'）==========
        # 'self_consumption_ratio': 0.85,  # 自用比例 (0-1)
        # 'retail_price': 0.90,            # 零售电价 (元/kWh)
        # 'feedin_price': 0.42,            # 余电上网电价 (元/kWh)

        # ========== 融资参数 ==========
        'capital_ratio': 0.25,            # 资本金比例 (0-1)
        'loan_rate': 0.048,              # 贷款利率

        # ========== 其他参数 ==========
        'deductible_tax': 400.0,          # 可抵扣进项税 (万元)，可选
    }


# ==============================================================================
# 主程序
# ==============================================================================

def main():
    print("=" * 80)
    print("🌟 PyPV-Eval v1.1.0 - 光伏项目技经评价完整示例")
    print("=" * 80)

    # 1. 获取项目参数
    params = get_project_params()

    print("\n📋 项目参数:")
    print(f"   装机容量: {params['capacity_mw']} MW")
    print(f"   静态投资: {params['static_invest']} 万元")
    print(f"   收益模式: {'全额上网' if params.get('mode') != 'self_consumption' else '自发自用'}")

    if params.get('mode') == 'self_consumption':
        print(f"   自用比例: {params.get('self_consumption_ratio', 0):.0%}")
        print(f"   零售电价: {params.get('retail_price', 0):.2f} 元/kWh")
        print(f"   上网电价: {params.get('feedin_price', 0):.2f} 元/kWh")
    else:
        print(f"   上网电价: {params.get('price_tax_inc', 0):.2f} 元/kWh")

    print(f"   利用小时: {params.get('hours', 0)} h")
    print(f"   资本金比例: {params.get('capital_ratio', 0):.0%}")

    # 2. 创建项目并计算现金流
    print("\n🔬 正在计算现金流...")
    project = PVProject(params)
    project.calculate_cash_flow()
    metrics = project.get_metrics()

    # 3. 输出核心指标
    print("\n" + "=" * 80)
    print("📊 核心财务指标")
    print("=" * 80)
    print(f"💰 项目总投资:       {metrics['总投资']:>15,.2f} 万元")
    print(f"🏗️  建设期利息:      {metrics['建设期利息']:>15,.2f} 万元")
    print(f"📈 全投资IRR(税前):  {metrics['全投资IRR(税前)']:>15,.2f} %")
    print(f"📉 全投资IRR(税后):  {metrics['全投资IRR(税后)']:>15,.2f} %")
    print(f"📅 投资回收期:       {metrics['投资回收期(年)']:>15,.2f} 年")

    # 4. 导出财务报表
    print("\n" + "=" * 80)
    print("📄 正在生成财务报表...")
    print("=" * 80)

    # 4.1 收入和税金表
    revenue_df = project.export_revenue_tax_table('output_收入和税金表.csv')
    print("✅ 收入和税金表: output_收入和税金表.csv")
    print(f"   25年总收入(含税): {revenue_df['营业收入(含税,万元)'].sum():,.2f} 万元")

    # 4.2 总成本费用表
    cost_df = project.export_total_cost_table('output_总成本费用表.csv')
    print("✅ 总成本费用表: output_总成本费用表.csv")
    print(f"   25年总成本: {cost_df['总成本费用(万元)'].sum():,.2f} 万元")

    # 4.3 利润与利润分配表
    profit_df = project.export_profit_table('output_利润表.csv')
    print("✅ 利润与利润分配表: output_利润表.csv")
    print(f"   25年累计净利润: {profit_df['累计净利润(万元)'].iloc[-1]:,.2f} 万元")

    # 4.4 项目总投资使用计划与资金筹措表
    investment_df = project.export_investment_plan_table('output_投资计划表.csv')
    print("✅ 项目总投资使用计划与资金筹措表: output_投资计划表.csv")

    # 5. 敏感性分析
    print("\n" + "=" * 80)
    print("📈 正在进行敏感性分析...")
    print("=" * 80)

    # 选择要分析的因素
    if params.get('mode') == 'self_consumption':
        factors = ['static_invest', 'hours', 'retail_price', 'feedin_price', 'self_consumption_ratio']
        factor_names = ['静态投资', '利用小时数', '零售电价', '上网电价', '自用比例']
    else:
        factors = ['static_invest', 'hours', 'price_tax_inc']
        factor_names = ['静态投资', '利用小时数', '上网电价']

    for factor, name in zip(factors, factor_names):
        # 修改参数名映射
        analysis_params = params.copy()
        if factor == 'price_tax_inc':
            analysis_params['price_tax_inc'] = params.get('price_tax_inc', 0.38)

        sens_df = sensitivity_analysis(analysis_params, factor, variation_range=0.15, steps=5)
        filename = f'output_敏感性分析_{name}.csv'
        sens_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"✅ {name}敏感性分析: {filename}")

        # 输出关键信息
        if not sens_df['IRR(税前)%'].isna().all():
            base_irr = sens_df.loc[sens_df['变化率'] == '0.0%', 'IRR(税前)%'].values[0] if '0.0%' in sens_df['变化率'].values else sens_df['IRR(税前)%'].iloc[len(sens_df)//2]
            max_var = sens_df['IRR(税前)%'].max()
            min_var = sens_df['IRR(税前)%'].min()
            print(f"   基准IRR: {base_irr:.2f}%, 范围: [{min_var:.2f}%, {max_var:.2f}%]")

    # 6. 反向求解（目标IRR推算投资）
    print("\n" + "=" * 80)
    print("🔮 反向求解（目标IRR -> 最大投资）")
    print("=" * 80)

    target_irr = 8.0
    max_invest = goal_seek_investment(target_irr, params)
    if max_invest is not None:
        print(f"   若目标 IRR = {target_irr}%:")
        print(f"   👉 最大允许静态投资: {max_invest:,.2f} 万元")
        print(f"   👉 单位造价: {max_invest / params['capacity_mw'] * 10000:,.2f} 元/W")

    print("\n" + "=" * 80)
    print("✅ 计算完成！所有报表已保存到当前目录")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
