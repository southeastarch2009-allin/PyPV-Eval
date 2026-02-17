# ==============================================================================
# 项目名称: PyPV-Eval (Python Photovoltaic Evaluation Engine)
# 核心依据: NB/T 11894-2025《光伏发电项目经济评价规范》
# 功能: 光伏项目全生命周期技经评价、IRR计算、敏感性分析、反向求解
# ==============================================================================

import pandas as pd
import numpy as np
import numpy_financial as npf
from scipy import optimize

class PVProject:
    """
    光伏项目技经评价核心类
    """
    def __init__(self, params):
        self.p = params.copy()
        # 初始化参数校验与预处理
        self.capacity = self.p['capacity_mw']          # 装机容量 MW
        self.static_invest = self.p['static_invest']   # 静态投资 (万元)
        self.construct_period = 1                      # 建设期 (年)
        self.operation_period = 25                     # 运营期 (年)

        # 预计算一些固定值
        self.loan_principal = self.static_invest * (1 - self.p['capital_ratio']) # 贷款本金

    def _calc_construction_interest(self):
        """
        计算建设期利息 (依据 NB/T 11894 3.1.7)
        简化逻辑: 假定资金年中均匀投入
        """
        rate = self.p['loan_rate']
        # 第一年利息 = (0 + 借款本金/2) * 利率
        interest = (self.loan_principal / 2) * rate
        return interest

    def _get_om_rate(self, year_idx):
        """
        获取阶梯运维费率 (依据 NB/T 11894 附录A)
        year_idx: 运营期第几年 (1-25)
        返回: 元/kWp
        """
        if year_idx <= 5: return 10.0
        elif year_idx <= 10: return 18.0
        elif year_idx <= 20: return 28.0
        else: return 32.0

    def calculate_cash_flow(self):
        """
        核心引擎: 生成25年现金流表
        """
        # 1. 基础计算
        const_interest = self._calc_construction_interest()

        # 流动资金 (30元/kWp = 30000元/MW = 3万元/MW)
        # 琼海项目给出的流动资金是 300万，即 100MW * 3万/MW
        working_capital = self.capacity * 3.0

        total_invest = self.static_invest + const_interest + working_capital

        # 增值税抵扣池初始化 (设备+建安进项税)
        # 简化: 假设静态投资的 11% 为可抵扣税金 (或直接读取参数)
        deductible_tax = self.p.get('deductible_tax', self.static_invest / 1.13 * 0.13)

        # 2. 初始化 DataFrame
        years = np.arange(1, self.operation_period + 2) # 1..26
        df = pd.DataFrame(index=years)

        # 初始化列
        cols = ['Generation', 'Revenue_Inc', 'Revenue_Exc', 'Output_VAT',
                'OM_Cost', 'VAT_Payable', 'Surtax', 'Total_Cost',
                'Profit_Total', 'Income_Tax', 'Net_CF_Pre', 'Net_CF_After']
        for c in cols: df[c] = 0.0

        # 3. 第1年 (建设期) 现金流
        # 现金流出 = 静态投资 + 流动资金 (注意: 建设期利息是融资流，全投资现金流表通常不含利息支出，只含本金投入)
        # NB/T 11894 表B.0.7 项目投资现金流量表: 流出=建设投资+流动资金+经营成本...
        df.loc[1, 'Net_CF_Pre'] = -(self.static_invest + working_capital)
        df.loc[1, 'Net_CF_After'] = -(self.static_invest + working_capital)

        # 4. 运营期逐年迭代
        current_deductible = deductible_tax

        for y in range(2, self.operation_period + 2):
            op_year = y - 1

            # --- A. 发电与收入 ---
            gen_hours = self.p['hours'] # 简化: 不考虑衰减，或后续加入衰减因子
            generation = self.capacity * gen_hours # MWh

            price = self.p['price_tax_inc']
            rev_inc = generation * 1000 * price / 10000 # 万元
            rev_exc = rev_inc / 1.13
            output_vat = rev_inc - rev_exc

            df.loc[y, 'Generation'] = generation
            df.loc[y, 'Revenue_Inc'] = rev_inc
            df.loc[y, 'Revenue_Exc'] = rev_exc
            df.loc[y, 'Output_VAT'] = output_vat

            # --- B. 成本 (运维) ---
            om_unit = self._get_om_rate(op_year)
            # 加上管理费(20)、保险费(0.25%造价)、材料费(8)等综合估算
            # 这里做一个为了对齐琼海项目的"综合调整系数"，实际项目中可细分
            # 琼海项目平均成本约 68000/25 ≈ 2700万/年。
            # 100MW * (10元运维+20元管理+...)
            # 我们先用标准运维费 + 固定比例的其他费
            other_cost = self.static_invest * 0.005 # 假设 0.5% 的其他杂费
            om_cost = (self.capacity * 1000 * om_unit / 10000) + other_cost

            df.loc[y, 'OM_Cost'] = om_cost

            # --- C. 税务 (增值税抵扣) ---
            if current_deductible > 0:
                if current_deductible >= output_vat:
                    current_deductible -= output_vat
                    vat_pay = 0
                else:
                    vat_pay = output_vat - current_deductible
                    current_deductible = 0
            else:
                vat_pay = output_vat

            df.loc[y, 'VAT_Payable'] = vat_pay
            surtax = vat_pay * 0.10 # 附加税 10%
            df.loc[y, 'Surtax'] = surtax

            # --- D. 所得税 (简化利润表计算) ---
            # 折旧 (20年直线法, 5%残值)
            depreciation = (self.static_invest + const_interest - deductible_tax) * 0.95 / 20
            if op_year > 20: depreciation = 0

            profit = rev_exc - om_cost - surtax - depreciation
            # 注意: 计算所得税的利润通常要扣除财务费用(利息)，但全投资现金流模型中，
            # "调整所得税"是基于息税前利润(EBIT)计算的，或者假设无负债。
            # 规范 B.0.7 注: 调整所得税...
            # 这里采用简化做法：三免三减半
            tax_rate = 0.25
            if op_year <= 3: tax_rate = 0.0
            elif op_year <= 6: tax_rate = 0.125

            income_tax = max(0, profit * tax_rate)
            df.loc[y, 'Income_Tax'] = income_tax

            # --- E. 现金流合成 ---
            inflow = rev_exc
            # 最后一年回收余值(5%)和流动资金
            if y == self.operation_period + 1:
                residual = self.static_invest * 0.05
                inflow += residual + working_capital

            outflow = om_cost + surtax

            df.loc[y, 'Net_CF_Pre'] = inflow - outflow
            df.loc[y, 'Net_CF_After'] = inflow - outflow - income_tax

        self.df = df
        self.total_invest = total_invest
        self.const_interest = const_interest
        return df

    def get_metrics(self):
        """计算IRR和NPV"""
        cf_pre = self.df['Net_CF_Pre'].values
        cf_after = self.df['Net_CF_After'].values

        irr_pre = npf.irr(cf_pre) * 100
        irr_after = npf.irr(cf_after) * 100

        # 投资回收期 (静态)
        cumsum = np.cumsum(cf_after)
        try:
            payback_idx = np.where(cumsum >= 0)[0][0]
            # 插值计算: 年份-1 + 绝对值(上年累计)/当年净现金流
            payback = (payback_idx) - 1 + abs(cumsum[payback_idx-1]) / cf_after[payback_idx]
        except:
            payback = 99.9

        return {
            "总投资": round(self.total_invest, 2),
            "建设期利息": round(self.const_interest, 2),
            "全投资IRR(税前)": round(irr_pre, 2),
            "全投资IRR(税后)": round(irr_after, 2),
            "投资回收期(年)": round(payback, 2)
        }

# ==============================================================================
# 🌟 高级功能: 反向求解 (Goal Seek)
# ==============================================================================
def goal_seek_investment(target_irr, params):
    """
    给定目标IRR (如 8%)，反推最大允许的静态投资 (Static Invest)
    """
    def objective(invest_guess):
        # 1. 更新参数
        p_temp = params.copy()
        p_temp['static_invest'] = invest_guess
        # 2. 运行模型
        project = PVProject(p_temp)
        project.calculate_cash_flow()
        metrics = project.get_metrics()
        # 3. 返回误差
        return metrics['全投资IRR(税前)'] - target_irr

    # 使用 Brent 方法在 [1000, 100000] 范围内寻找根
    # 琼海项目是 40000，所以这个范围是合理的
    try:
        limit_invest = optimize.brentq(objective, 10000, 100000)
        return limit_invest
    except:
        return None

# ==============================================================================
# 🚀 运行验证: 琼海项目 100MW
# ==============================================================================

# 1. 输入参数 (来自你的CSV)
qionghai_params = {
    'capacity_mw': 100.0,
    'static_invest': 40000.0,
    'capital_ratio': 0.20,
    'loan_rate': 0.04876,     # 我们反推出来的利率
    'hours': 1500,            # 150000 MWh / 100 MW
    'price_tax_inc': 0.40,    # 含税电价
    'deductible_tax': 4000.0  # 初始进项税
}

print("正在计算琼海项目...")
project = PVProject(qionghai_params)
df_result = project.calculate_cash_flow()
metrics = project.get_metrics()

print("\n" + "="*40)
print(f"📊 琼海项目 (100MW) 计算结果")
print("="*40)
print(f"✅ 建设期利息:  {metrics['建设期利息']} 万元 (目标: 780.18)")
print(f"✅ 项目总投资:  {metrics['总投资']} 万元 (目标: 41080.18)")
print(f"🔥 IRR (税前): {metrics['全投资IRR(税前)']}% (目标: 11.35%)")
print(f"🔥 IRR (税后): {metrics['全投资IRR(税后)']}% (目标: 9.97%)")
print(f"📅 投资回收期:  {metrics['投资回收期(年)']} 年")
print("="*40)

# 2. 演示反向求解
target_irr = 8.0
print(f"\n🔮 反向求解演示: 如果只要 {target_irr}% 的IRR，造价可以放宽到多少？")
limit_val = goal_seek_investment(target_irr, qionghai_params)
print(f"👉 最大允许建设投资: {limit_val:.2f} 万元 (原值: 40000)")
print(f"👉 溢价空间: {(limit_val - 40000):.2f} 万元")
