# ==============================================================================
# 项目名称: PyPV-Eval (Python Photovoltaic Evaluation Engine)
# 版本: v1.1.0 (Enhanced - 支持自发自用模式)
# 核心依据: NB/T 11894-2025《光伏发电项目经济评价规范》
# ==============================================================================

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import numpy_financial as npf
from scipy import optimize

# ==============================================================================
# 常量定义 (提取魔法数字)
# ==============================================================================

class Constants:
    """项目常量配置"""

    # 收益模式
    MODE_FULL_GRID = 'full_grid'              # 全额上网模式
    MODE_SELF_CONSUMPTION = 'self_consumption'  # 自发自用、余额上网模式

    # 税率相关
    VAT_RATE = 0.13              # 增值税率 13%
    SURTAX_RATE = 0.10           # 附加税率 10% (城建7%+教育3%)
    INCOME_TAX_RATE = 0.25       # 企业所得税率 25%

    # 折旧相关
    DEPRECIATION_YEARS = 20      # 折旧年限
    RESIDUAL_RATIO = 0.05        # 残值率 5%
    DEPRECIATION_BASE_RATIO = 0.95  # 折旧基数比例 (1 - 残值率)

    # 项目期限
    CONSTRUCT_PERIOD = 1         # 建设期 (年)
    OPERATION_PERIOD = 25        # 运营期 (年)

    # 费用相关
    WORKING_CAPITAL_PER_MW = 3.0  # 流动资金 (万元/MW)
    OTHER_COST_RATIO = 0.005      # 其他费用比例 0.5%

    # 运维费率 (元/kWp) - NB/T 11894 附录A 表A.1.1
    OM_RATES = {
        (1, 5): 10.0,    # 1-5年
        (6, 10): 18.0,   # 6-10年
        (11, 20): 28.0,  # 11-20年
        (21, 25): 32.0   # 21-25年
    }

    # Goal Seek 求解范围
    MIN_INVEST = 1000     # 最小投资 (万元)
    MAX_INVEST = 100000   # 最大投资 (万元)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# 异常定义
# ==============================================================================

class PVProjectError(Exception):
    """光伏项目评价基础异常"""
    pass


class InputValidationError(PVProjectError):
    """输入参数验证失败异常"""
    pass


class CalculationError(PVProjectError):
    """计算失败异常"""
    pass


# ==============================================================================
# 核心类
# ==============================================================================

class PVProject:
    """
    光伏项目技经评价核心类

    严格遵循 NB/T 11894-2025《光伏发电项目经济评价规范》

    支持两种收益模式:
        1. 全额上网 (full_grid): 全部发电量按上网电价销售
        2. 自发自用 (self_consumption): 自用部分节省购电成本，余电上网销售

    Attributes:
        capacity: 装机容量 (MW)
        static_invest: 静态投资 (万元)
        gen_hours: 年利用小时数 (h)
        loan_rate: 长期贷款利率
        capital_ratio: 资本金比例
        mode: 收益模式 ('full_grid' 或 'self_consumption')
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        初始化光伏项目

        Args:
            params: 项目参数字典，包含以下键:

            通用参数:
                - capacity_mw: 装机容量 (MW)
                - static_invest: 静态投资 (万元)
                - hours: 年利用小时数 (h)，默认1000
                - loan_rate: 长期贷款利率，默认0.049
                - capital_ratio: 资本金比例，默认0.2
                - mode: 收益模式，'full_grid'(默认) 或 'self_consumption'
                - deductible_tax: 可抵扣进项税 (万元)，可选

            全额上网模式 (mode='full_grid'):
                - price_tax_inc: 含税上网电价 (元/kWh)

            自发自用模式 (mode='self_consumption'):
                - self_consumption_ratio: 自用比例 (0-1)，如0.8表示80%自用
                - retail_price: 零售电价/工商业电价 (元/kWh)，自用节省的单价
                - feedin_price: 余电上网电价 (元/kWh)，余电销售的单价

        Raises:
            InputValidationError: 参数验证失败
        """
        self.p = params.copy()
        self._validate_and_init_params()
        self.df: Optional[pd.DataFrame] = None
        self.total_invest: float = 0.0
        self.const_interest: float = 0.0

    def _validate_and_init_params(self) -> None:
        """参数校验与标准化"""
        # 获取模式参数，默认为全额上网
        self.mode = self.p.get('mode', Constants.MODE_FULL_GRID)

        # 验证模式参数
        if self.mode not in [Constants.MODE_FULL_GRID, Constants.MODE_SELF_CONSUMPTION]:
            raise InputValidationError(
                f"无效的 mode 参数: {self.mode}。"
                f"必须是 '{Constants.MODE_FULL_GRID}' 或 '{Constants.MODE_SELF_CONSUMPTION}'"
            )

        # 验证通用必需参数
        required_keys = ['capacity_mw', 'static_invest']
        missing_keys = [k for k in required_keys if k not in self.p]
        if missing_keys:
            raise InputValidationError(f"缺少必需参数: {missing_keys}")

        # 获取并验证通用参数
        self.capacity = float(self.p.get('capacity_mw', 0))
        self.static_invest = float(self.p.get('static_invest', 0))
        self.gen_hours = float(self.p.get('hours', 1000))
        self.loan_rate = float(self.p.get('loan_rate', 0.049))
        self.capital_ratio = float(self.p.get('capital_ratio', 0.2))

        # 数值范围验证
        if self.capacity <= 0:
            raise InputValidationError("装机容量必须大于0")
        if self.static_invest <= 0:
            raise InputValidationError("静态投资必须大于0")
        if self.gen_hours <= 0:
            raise InputValidationError("年利用小时数必须大于0")
        if not 0 < self.capital_ratio <= 1:
            raise InputValidationError("资本金比例必须在 (0, 1] 范围内")

        # 预计算贷款本金
        self.loan_principal = self.static_invest * (1 - self.capital_ratio)

        # 根据模式验证特定参数
        if self.mode == Constants.MODE_FULL_GRID:
            if 'price_tax_inc' not in self.p:
                raise InputValidationError("全额上网模式需要参数: price_tax_inc")
            self.price_tax_inc = float(self.p['price_tax_inc'])
            logger.info(f"模式: 全额上网, 电价={self.price_tax_inc}元/kWh")

        elif self.mode == Constants.MODE_SELF_CONSUMPTION:
            required_sc_keys = ['self_consumption_ratio', 'retail_price', 'feedin_price']
            missing_sc_keys = [k for k in required_sc_keys if k not in self.p]
            if missing_sc_keys:
                raise InputValidationError(f"自发自用模式需要参数: {missing_sc_keys}")

            self.self_consumption_ratio = float(self.p['self_consumption_ratio'])
            self.retail_price = float(self.p['retail_price'])
            self.feedin_price = float(self.p['feedin_price'])

            # 验证自用比例
            if not 0 <= self.self_consumption_ratio <= 1:
                raise InputValidationError("自用比例必须在 [0, 1] 范围内")

            logger.info(
                f"模式: 自发自用, 自用比例={self.self_consumption_ratio:.1%}, "
                f"零售电价={self.retail_price}元/kWh, 上网电价={self.feedin_price}元/kWh"
            )

        logger.info(f"项目参数验证通过: 容量={self.capacity}MW, 投资={self.static_invest}万元")

    def _calc_construction_interest(self) -> float:
        """
        计算建设期利息

        依据: NB/T 11894 3.1.7
        简化逻辑: 假定资金年中均匀投入
        公式: Construction Interest = (Loan / 2) * Rate

        Returns:
            建设期利息 (万元)
        """
        interest = (self.loan_principal / 2) * self.loan_rate
        return interest

    def _get_om_rate(self, year_idx: int) -> float:
        """
        获取阶梯运维费率

        依据: NB/T 11894 附录A 表A.1.1

        Args:
            year_idx: 运营期第几年 (1-25)

        Returns:
            运维费率 (元/kWp)
        """
        for (start, end), rate in Constants.OM_RATES.items():
            if start <= year_idx <= end:
                return rate
        return Constants.OM_RATES[(21, 25)]  # 默认返回最高档

    def calculate_cash_flow(self) -> pd.DataFrame:
        """
        核心引擎: 生成25年现金流表

        Returns:
            包含完整现金流数据的DataFrame

        Raises:
            CalculationError: 计算过程中发生错误
        """
        try:
            # --- A. 建设期计算 ---
            const_interest = self._calc_construction_interest()
            working_capital = self.capacity * Constants.WORKING_CAPITAL_PER_MW
            total_invest = self.static_invest + const_interest + working_capital

            # 增值税抵扣池初始化 (依据 NB/T 11894 3.2.6)
            deductible_tax = self.p.get(
                'deductible_tax',
                self.static_invest / (1 + Constants.VAT_RATE) * Constants.VAT_RATE
            )

            # --- B. 初始化现金流表 ---
            years = np.arange(1, Constants.OPERATION_PERIOD + 2)
            df = pd.DataFrame(index=years)

            cols = [
                'Generation', 'Revenue_Inc', 'Revenue_Exc', 'Output_VAT',
                'OM_Cost', 'VAT_Payable', 'Surtax', 'Total_Cost',
                'Profit_Total', 'Income_Tax', 'Net_CF_Pre', 'Net_CF_After'
            ]
            for c in cols:
                df[c] = 0.0

            # 第1年 (建设期) 现金流出
            df.loc[1, 'Net_CF_Pre'] = -(self.static_invest + working_capital)
            df.loc[1, 'Net_CF_After'] = -(self.static_invest + working_capital)

            # --- C. 运营期逐年迭代 ---
            current_deductible = deductible_tax
            fixed_asset_value = self.static_invest + const_interest - deductible_tax

            for y in range(2, Constants.OPERATION_PERIOD + 2):
                op_year = y - 1

                # 1. 发电与收入计算
                generation = self.capacity * self.gen_hours  # MWh

                if self.mode == Constants.MODE_FULL_GRID:
                    # 全额上网模式：全部发电量按上网电价计算
                    price = self.price_tax_inc
                    rev_inc = generation * 1000 * price / 10000  # 万元
                    rev_exc = rev_inc / (1 + Constants.VAT_RATE)
                    output_vat = rev_inc - rev_exc

                else:  # MODE_SELF_CONSUMPTION
                    # 自发自用模式：拆分为自用和余电两部分
                    self_consumed_mwh = generation * self.self_consumption_ratio
                    surplus_mwh = generation * (1 - self.self_consumption_ratio)

                    # 自用部分收益 = 避免购电的成本节省（按零售电价）
                    # 注意：自用节省是否涉及VAT处理取决于具体政策
                    # 这里简化处理：自用部分按不含税零售价计算收益
                    rev_self_exc = self_consumed_mwh * 1000 * self.retail_price / 10000 / (1 + Constants.VAT_RATE)

                    # 余电上网收益 = 余电 × 上网电价
                    rev_surplus_inc = surplus_mwh * 1000 * self.feedin_price / 10000
                    rev_surplus_exc = rev_surplus_inc / (1 + Constants.VAT_RATE)
                    vat_surplus = rev_surplus_inc - rev_surplus_exc

                    # 总收益
                    rev_inc = rev_surplus_inc  # 增值税基数只有余电上网部分
                    rev_exc = rev_self_exc + rev_surplus_exc
                    output_vat = vat_surplus  # 只有余电上网部分产生销项税

                    logger.debug(
                        f"第{op_year}年: 发电={generation:.1f}MWh, "
                        f"自用={self_consumed_mwh:.1f}MWh, 余电={surplus_mwh:.1f}MWh"
                    )

                df.loc[y, 'Generation'] = generation
                df.loc[y, 'Revenue_Inc'] = rev_inc
                df.loc[y, 'Revenue_Exc'] = rev_exc
                df.loc[y, 'Output_VAT'] = output_vat

                # 2. 成本 (运维 + 其他)
                om_unit = self._get_om_rate(op_year)
                om_cost = (
                    self.capacity * 1000 * om_unit / 10000
                    + self.static_invest * Constants.OTHER_COST_RATIO
                )
                df.loc[y, 'OM_Cost'] = om_cost

                # 3. 税务 (增值税抵扣池逻辑)
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
                surtax = vat_pay * Constants.SURTAX_RATE
                df.loc[y, 'Surtax'] = surtax

                # 4. 利润与所得税
                depreciation = (
                    fixed_asset_value * Constants.DEPRECIATION_BASE_RATIO / Constants.DEPRECIATION_YEARS
                    if op_year <= Constants.DEPRECIATION_YEARS else 0
                )

                profit = rev_exc - om_cost - surtax - depreciation

                # 三免三减半政策
                if op_year <= 3:
                    tax_rate = 0.0
                elif op_year <= 6:
                    tax_rate = Constants.INCOME_TAX_RATE * 0.5
                else:
                    tax_rate = Constants.INCOME_TAX_RATE

                income_tax = max(0.0, profit * tax_rate)
                df.loc[y, 'Income_Tax'] = income_tax

                # 5. 现金流合成
                inflow = rev_exc
                if y == Constants.OPERATION_PERIOD + 1:
                    residual = self.static_invest * Constants.RESIDUAL_RATIO
                    inflow += residual + working_capital

                outflow = om_cost + surtax
                df.loc[y, 'Net_CF_Pre'] = inflow - outflow
                df.loc[y, 'Net_CF_After'] = inflow - outflow - income_tax

            self.df = df
            self.total_invest = total_invest
            self.const_interest = const_interest

            logger.info(f"现金流计算完成: 总投资={total_invest:.2f}万元")
            return df

        except Exception as e:
            raise CalculationError(f"现金流计算失败: {e}") from e

    def get_metrics(self) -> Dict[str, float]:
        """
        计算核心指标

        Returns:
            包含以下指标的字典:
                - 总投资 (万元)
                - 建设期利息 (万元)
                - 全投资IRR(税前) (%)
                - 全投资IRR(税后) (%)
                - 投资回收期 (年)

        Raises:
            CalculationError: 指标计算失败
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        try:
            cf_pre = self.df['Net_CF_Pre'].values
            cf_after = self.df['Net_CF_After'].values

            irr_pre = npf.irr(cf_pre) * 100
            irr_after = npf.irr(cf_after) * 100

            # 静态投资回收期计算
            cumsum = np.cumsum(cf_after)
            positive_indices = np.where(cumsum >= 0)[0]

            if len(positive_indices) > 0:
                p_idx = positive_indices[0]
                payback = p_idx - 1 + abs(cumsum[p_idx - 1]) / cf_after[p_idx] if p_idx > 0 else 1.0
            else:
                logger.warning("项目在运营期内无法收回投资")
                payback = 99.9

            return {
                "总投资": round(self.total_invest, 2),
                "建设期利息": round(self.const_interest, 2),
                "全投资IRR(税前)": round(irr_pre, 2),
                "全投资IRR(税后)": round(irr_after, 2),
                "投资回收期(年)": round(payback, 2)
            }

        except Exception as e:
            raise CalculationError(f"指标计算失败: {e}") from e

    # ==============================================================================
    # 财务报表输出方法
    # ==============================================================================

    def export_revenue_tax_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出收入和税金表

        依据 NB/T 11894-2025 表 B.0.3

        Args:
            filename: 输出文件名，如 'revenue_tax.csv'，为 None 则返回 DataFrame

        Returns:
            收入和税金表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        # 提取运营期数据
        df = self.df[self.df.index >= 2].copy()

        # 创建收入和税金表
        table = pd.DataFrame({
            '年份': [f'第{i}年' for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '发电量(MWh)': df['Generation'].values,
            '营业收入(含税,万元)': df['Revenue_Inc'].values,
            '营业收入(不含税,万元)': df['Revenue_Exc'].values,
            '增值税(万元)': df['Output_VAT'].values,
            '增值税实缴(万元)': df['VAT_Payable'].values,
            '附加税(万元)': df['Surtax'].values,
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"收入和税金表已保存到: {filename}")

        return table

    def export_total_cost_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出总成本费用估算表

        依据 NB/T 11894-2025 表 B.0.5

        Args:
            filename: 输出文件名，如 'total_cost.csv'

        Returns:
            总成本费用表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        df = self.df[self.df.index >= 2].copy()

        # 重新计算折旧
        deductible_tax = self.p.get(
            'deductible_tax',
            self.static_invest / (1 + Constants.VAT_RATE) * Constants.VAT_RATE
        )
        const_interest = self.const_interest
        fixed_asset_value = self.static_invest + const_interest - deductible_tax

        depreciation_per_year = fixed_asset_value * Constants.DEPRECIATION_BASE_RATIO / Constants.DEPRECIATION_YEARS

        # 创建总成本费用表
        table = pd.DataFrame({
            '年份': [f'第{i}年' for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '运维成本(万元)': df['OM_Cost'].values,
            '折旧费(万元)': [depreciation_per_year if i <= Constants.DEPRECIATION_YEARS else 0
                            for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '摊销费(万元)': [0.0] * Constants.OPERATION_PERIOD,
            '财务费用(万元)': [0.0] * Constants.OPERATION_PERIOD,  # 融资前分析
            '总成本费用(万元)': df['OM_Cost'].values + [depreciation_per_year if i <= Constants.DEPRECIATION_YEARS else 0
                            for i in range(1, Constants.OPERATION_PERIOD + 1)],
        })

        # 经营成本 = 总成本 - 折旧 - 摊销 - 财务费用
        table['经营成本(万元)'] = table['运维成本(万元)']

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"总成本费用表已保存到: {filename}")

        return table

    def export_profit_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出利润与利润分配表

        依据 NB/T 11894-2025 表 B.0.6

        Args:
            filename: 输出文件名，如 'profit.csv'

        Returns:
            利润表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        df = self.df[self.df.index >= 2].copy()

        # 重新计算折旧
        deductible_tax = self.p.get(
            'deductible_tax',
            self.static_invest / (1 + Constants.VAT_RATE) * Constants.VAT_RATE
        )
        const_interest = self.const_interest
        fixed_asset_value = self.static_invest + const_interest - deductible_tax
        depreciation_per_year = fixed_asset_value * Constants.DEPRECIATION_BASE_RATIO / Constants.DEPRECIATION_YEARS

        # 计算利润
        profit_list = []
        for i in range(1, Constants.OPERATION_PERIOD + 1):
            depreciation = depreciation_per_year if i <= Constants.DEPRECIATION_YEARS else 0
            profit = df.loc[i + 1, 'Revenue_Exc'] - df.loc[i + 1, 'OM_Cost'] - df.loc[i + 1, 'Surtax'] - depreciation
            profit_list.append(profit)

        # 创建利润表
        table = pd.DataFrame({
            '年份': [f'第{i}年' for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '营业收入(不含税,万元)': df['Revenue_Exc'].values,
            '营业税金及附加(万元)': df['Surtax'].values,
            '总成本费用(万元)': df['OM_Cost'].values + [depreciation_per_year if i <= Constants.DEPRECIATION_YEARS else 0
                            for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '利润总额(万元)': profit_list,
            '所得税(万元)': df['Income_Tax'].values,
            '净利润(万元)': [p - t for p, t in zip(profit_list, df['Income_Tax'].values)],
        })

        # 累计净利润
        table['累计净利润(万元)'] = table['净利润(万元)'].cumsum()

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"利润表已保存到: {filename}")

        return table

    def export_investment_plan_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出项目总投资使用计划与资金筹措表

        依据 NB/T 11894-2025 表 B.0.2

        Args:
            filename: 输出文件名，如 'investment_plan.csv'

        Returns:
            投资计划表 DataFrame
        """
        # 建设期利息
        const_interest = self.const_interest
        working_capital = self.capacity * Constants.WORKING_CAPITAL_PER_MW

        # 创建投资计划表
        table = pd.DataFrame({
            '项目': [
                '建设投资',
                '建设期利息',
                '流动资金',
                '项目总投资'
            ],
            '合计(万元)': [
                self.static_invest,
                const_interest,
                working_capital,
                self.total_invest
            ],
            '第1年(万元)': [
                self.static_invest,
                const_interest,
                working_capital,
                self.static_invest + const_interest + working_capital
            ]
        })

        # 资本金和银行贷款
        capital_amount = self.static_invest * self.capital_ratio
        loan_amount = self.static_invest * (1 - self.capital_ratio)
        loan_with_interest = loan_amount + const_interest

        # 添加筹措部分
        funding_df = pd.DataFrame({
            '项目': [
                '项目资本金',
                '银行贷款',
                '资金筹措合计'
            ],
            '合计(万元)': [
                capital_amount + working_capital * self.capital_ratio,  # 假设流动资金也按相同比例
                loan_with_interest + working_capital * (1 - self.capital_ratio),
                capital_amount + working_capital + loan_with_interest
            ],
            '第1年(万元)': [
                capital_amount + working_capital * self.capital_ratio,
                loan_with_interest + working_capital * (1 - self.capital_ratio),
                capital_amount + working_capital + loan_with_interest
            ]
        })

        table = pd.concat([table, pd.DataFrame({'项目': [''], '合计(万元)': [''], '第1年(万元)': ['']})], ignore_index=True)
        table = pd.concat([table, funding_df], ignore_index=True)

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"投资计划表已保存到: {filename}")

        return table

    def export_financial_cash_flow_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出财务现金流量表（融资后分析）

        依据 NB/T 11894-2025 表 B.0.9

        Args:
            filename: 输出文件名，如 'financial_cashflow.csv'

        Returns:
            财务现金流量表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        df = self.df.copy()

        # 创建财务现金流量表
        table = pd.DataFrame({
            '年份': ['建设期'] + [f'第{i}年' for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '现金流入(万元)': [0] + list(df.loc[2:, 'Revenue_Exc'].values) + [df.loc[Constants.OPERATION_PERIOD + 1, 'Revenue_Exc'] +
                self.static_invest * Constants.RESIDUAL_RATIO + self.capacity * Constants.WORKING_CAPITAL_PER_MW],
            '现金流出(万元)': [df.loc[1, 'Net_CF_After']] + list(
                (df.loc[2:, 'OM_Cost'] + df.loc[2:, 'Surtax'] + df.loc[2:, 'Income_Tax'] +
                 df.loc[2:, 'OM_Cost'] * 0).values) + [df.loc[Constants.OPERATION_PERIOD + 1, 'OM_Cost'] +
                df.loc[Constants.OPERATION_PERIOD + 1, 'Surtax'] + df.loc[Constants.OPERATION_PERIOD + 1, 'Income_Tax']],
            '净现金流量(万元)': [df.loc[1, 'Net_CF_After']] + list(df.loc[2:, 'Net_CF_After'].values),
            '累计净现金流量(万元)': [df.loc[1, 'Net_CF_After']] + list(df.loc[2:, 'Net_CF_After'].cumsum().values),
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"财务现金流量表已保存到: {filename}")

        return table

    def export_project_investment_cashflow_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出项目投资现金流量表（融资前分析）

        依据 NB/T 11894-2025 表 B.0.7

        Args:
            filename: 输出文件名，如 'project_investment_cashflow.csv'

        Returns:
            项目投资现金流量表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        df = self.df.copy()

        table = pd.DataFrame({
            '年份': ['建设期'] + [f'第{i}年' for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '现金流入(万元)': [0] + list(df.loc[2:, 'Revenue_Exc'].values) + [df.loc[Constants.OPERATION_PERIOD + 1, 'Revenue_Exc'] +
                self.static_invest * Constants.RESIDUAL_RATIO + self.capacity * Constants.WORKING_CAPITAL_PER_MW],
            '现金流出(万元)': [df.loc[1, 'Net_CF_Pre']] + list(
                (df.loc[2:, 'OM_Cost'] + df.loc[2:, 'Surtax']).values) + [df.loc[Constants.OPERATION_PERIOD + 1, 'OM_Cost'] +
                df.loc[Constants.OPERATION_PERIOD + 1, 'Surtax']],
            '所得税前净现金流量(万元)': [df.loc[1, 'Net_CF_Pre']] + list(df.loc[2:, 'Net_CF_Pre'].values),
            '累计所得税前净现金流量(万元)': [df.loc[1, 'Net_CF_Pre']] + list(df.loc[2:, 'Net_CF_Pre'].cumsum().values),
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"项目投资现金流量表已保存到: {filename}")

        return table

    def export_capital_cashflow_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出项目资本金现金流量表

        依据 NB/T 11894-2025 表 B.0.8

        Args:
            filename: 输出文件名，如 'capital_cashflow.csv'

        Returns:
            资本金现金流量表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        df = self.df.copy()

        # 资本金投入
        capital_invest = self.static_invest * self.capital_ratio
        working_capital = self.capacity * Constants.WORKING_CAPITAL_PER_MW

        table = pd.DataFrame({
            '年份': ['建设期'] + [f'第{i}年' for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '现金流入(万元)': [0] + list(df.loc[2:, 'Net_CF_After'].values) + [df.loc[Constants.OPERATION_PERIOD + 1, 'Net_CF_After'] +
                self.static_invest * Constants.RESIDUAL_RATIO + working_capital],
            '资本金投入(万元)': [-(capital_invest + working_capital * self.capital_ratio)] + [0] * Constants.OPERATION_PERIOD,
            '借款本金偿还(万元)': [0] * (Constants.OPERATION_PERIOD + 1),
            '借款利息支付(万元)': [0] * (Constants.OPERATION_PERIOD + 1),
            '现金流出(万元)': [-(capital_invest + working_capital * self.capital_ratio)] + [0] * Constants.OPERATION_PERIOD,
            '净现金流量(万元)': [-(capital_invest + working_capital * self.capital_ratio)] + list(df.loc[2:, 'Net_CF_After'].values),
            '累计净现金流量(万元)': [-(capital_invest + working_capital * self.capital_ratio)] + list(df.loc[2:, 'Net_CF_After'].cumsum().values),
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"资本金现金流量表已保存到: {filename}")

        return table

    def export_balance_sheet(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出资产负债表

        依据 NB/T 11894-2025 表 B.0.10

        Args:
            filename: 输出文件名，如 'balance_sheet.csv'

        Returns:
            资产负债表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        # 计算累计利润和资产
        df = self.df[self.df.index >= 2].copy()

        # 重新计算折旧
        deductible_tax = self.p.get(
            'deductible_tax',
            self.static_invest / (1 + Constants.VAT_RATE) * Constants.VAT_RATE
        )
        const_interest = self.const_interest
        fixed_asset_value = self.static_invest + const_interest - deductible_tax
        depreciation_per_year = fixed_asset_value * Constants.DEPRECIATION_BASE_RATIO / Constants.DEPRECIATION_YEARS

        # 计算累计折旧
        accumulated_depreciation = []
        acc_dep = 0
        for i in range(1, Constants.OPERATION_PERIOD + 1):
            if i <= Constants.DEPRECIATION_YEARS:
                acc_dep += depreciation_per_year
            accumulated_depreciation.append(acc_dep)

        # 计算累计利润
        cumulative_profit = []
        cum_profit = 0
        for i in range(1, Constants.OPERATION_PERIOD + 1):
            depreciation = depreciation_per_year if i <= Constants.DEPRECIATION_YEARS else 0
            profit = df.loc[i + 1, 'Revenue_Exc'] - df.loc[i + 1, 'OM_Cost'] - df.loc[i + 1, 'Surtax'] - depreciation
            after_tax_profit = profit - df.loc[i + 1, 'Income_Tax']
            cum_profit += after_tax_profit
            cumulative_profit.append(cum_profit)

        table = pd.DataFrame({
            '年份': [f'第{i}年' for i in range(1, Constants.OPERATION_PERIOD + 1)],
            # 资产
            '流动资产总额(万元)': [self.capacity * Constants.WORKING_CAPITAL_PER_MW] * Constants.OPERATION_PERIOD,
            '固定资产净值(万元)': [fixed_asset_value - d for d in accumulated_depreciation],
            '资产总额(万元)': [self.capacity * Constants.WORKING_CAPITAL_PER_MW + fixed_asset_value - d for d in accumulated_depreciation],
            # 负债
            '流动负债(万元)': [0] * Constants.OPERATION_PERIOD,
            '长期借款(万元)': [0] * Constants.OPERATION_PERIOD,
            '负债合计(万元)': [0] * Constants.OPERATION_PERIOD,
            # 所有者权益
            '资本金(万元)': [self.static_invest * self.capital_ratio] * Constants.OPERATION_PERIOD,
            '累计盈余公积金(万元)': [max(0, p * 0.1) for p in cumulative_profit],  # 假设提取10%盈余公积
            '累计未分配利润(万元)': cumulative_profit,
            '所有者权益合计(万元)': [self.static_invest * self.capital_ratio + max(0, p * 0.1) + p for p in cumulative_profit],
            '负债及所有者权益(万元)': [self.static_invest * self.capital_ratio + max(0, p * 0.1) + p
                                      for p in cumulative_profit],
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"资产负债表已保存到: {filename}")

        return table

    def export_financial_summary_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出财务指标汇总表

        Args:
            filename: 输出文件名，如 'financial_summary.csv'

        Returns:
            财务指标汇总表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        metrics = self.get_metrics()

        # 计算更多财务指标
        df = self.df[self.df.index >= 2].copy()

        # 计算总投资收益率 (ROI)
        total_profit = df['Revenue_Exc'].sum() - df['OM_Cost'].sum() - df['Surtax'].sum()
        roi = total_profit / self.total_invest * 100

        # 计算投资利税率
        total_tax = df['VAT_Payable'].sum() + df['Surtax'].sum() + df['Income_Tax'].sum()
        investment_profit_tax_rate = (total_profit + total_tax) / self.total_invest * 100

        table = pd.DataFrame({
            '指标': [
                '项目总投资(万元)',
                '建设期利息(万元)',
                '全投资IRR(税前,%)',
                '全投资IRR(税后,%)',
                '投资回收期(年)',
                '总投资收益率(ROI,%)',
                '投资利税率(%)',
                '年均净利润(万元)',
                '25年累计净利润(万元)',
                '装机容量(MW)',
                '单位造价(元/W)',
            ],
            '数值': [
                metrics['总投资'],
                metrics['建设期利息'],
                metrics['全投资IRR(税前)'],
                metrics['全投资IRR(税后)'],
                metrics['投资回收期(年)'],
                round(roi, 2),
                round(investment_profit_tax_rate, 2),
                round(df['Revenue_Exc'].sum() - df['OM_Cost'].sum() - df['Surtax'].sum() - df['Income_Tax'].sum(), 2) / 25,
                round(df['Revenue_Exc'].sum() - df['OM_Cost'].sum() - df['Surtax'].sum() - df['Income_Tax'].sum(), 2),
                self.capacity,
                round(self.static_invest / self.capacity * 10000, 2),
            ],
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"财务指标汇总表已保存到: {filename}")

        return table

    def export_parameters_summary_table(self, filename: Optional[str] = None) -> pd.DataFrame:
        """
        导出参数汇总表

        Args:
            filename: 输出文件名，如 'parameters_summary.csv'

        Returns:
            参数汇总表 DataFrame
        """
        table = pd.DataFrame({
            '参数类别': [
                '基础参数',
                '基础参数',
                '基础参数',
                '基础参数',
                '收益模式',
                '电价参数',
                '电价参数',
                '电价参数',
                '融资参数',
                '融资参数',
                '税务参数',
            ],
            '参数名称': [
                '装机容量',
                '静态投资',
                '年利用小时数',
                '建设期(年)',
                '收益模式',
                '上网电价' if self.mode == Constants.MODE_FULL_GRID else '零售电价',
                '上网电价' if self.mode == Constants.MODE_FULL_GRID else '余电上网电价',
                '自用比例' if self.mode == Constants.MODE_SELF_CONSUMPTION else '-',
                '资本金比例',
                '贷款利率',
                '可抵扣进项税',
            ],
            '参数值': [
                f"{self.capacity} MW",
                f"{self.static_invest} 万元",
                f"{self.gen_hours} 小时",
                Constants.CONSTRUCT_PERIOD,
                '全额上网' if self.mode == Constants.MODE_FULL_GRID else '自发自用',
                f"{self.price_tax_inc if self.mode == Constants.MODE_FULL_GRID else self.retail_price} 元/kWh",
                f"-" if self.mode == Constants.MODE_FULL_GRID else f"{self.feedin_price} 元/kWh",
                f"{self.self_consumption_ratio:.1%}" if self.mode == Constants.MODE_SELF_CONSUMPTION else "-",
                f"{self.capital_ratio:.1%}",
                f"{self.loan_rate:.2%}",
                f"{self.p.get('deductible_tax', '自动计算')} 万元",
            ],
            '备注': [
                '-',
                '-',
                '-',
                '-',
                '-',
                '-',
                '-',
                '-',
                '-',
                '-',
                '按13%税率估算',
            ],
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"参数汇总表已保存到: {filename}")

        return table

    def export_eva_table(self, filename: Optional[str] = None, wacc: float = 0.06) -> pd.DataFrame:
        """
        导出EVA（经济增加值）测算表

        EVA = 税后净营业利润 - 资本成本

        Args:
            filename: 输出文件名，如 'eva_analysis.csv'
            wacc: 加权平均资本成本，默认6%

        Returns:
            EVA测算表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        df = self.df[self.df.index >= 2].copy()

        # 计算税后净营业利润 (NOPAT)
        deductible_tax = self.p.get(
            'deductible_tax',
            self.static_invest / (1 + Constants.VAT_RATE) * Constants.VAT_RATE
        )
        const_interest = self.const_interest
        fixed_asset_value = self.static_invest + const_interest - deductible_tax
        depreciation_per_year = fixed_asset_value * Constants.DEPRECIATION_BASE_RATIO / Constants.DEPRECIATION_YEARS

        nopat_list = []
        capital_list = []
        eva_list = []

        capital = self.total_invest  # 初始投资

        for i in range(1, Constants.OPERATION_PERIOD + 1):
            depreciation = depreciation_per_year if i <= Constants.DEPRECIATION_YEARS else 0

            # 税后净营业利润 = 净利润 + 利息支出(1-税率) + 所得税
            profit = df.loc[i + 1, 'Revenue_Exc'] - df.loc[i + 1, 'OM_Cost'] - df.loc[i + 1, 'Surtax'] - depreciation
            income_tax = df.loc[i + 1, 'Income_Tax']
            nopat = profit - income_tax

            nopat_list.append(nopat)

            # 资本占用 (年末)
            if i <= Constants.DEPRECIATION_YEARS:
                capital = capital - depreciation_per_year
            capital_list.append(capital)

            # EVA = NOPAT - 资本占用 × WACC
            eva = nopat - capital * wacc
            eva_list.append(eva)

        table = pd.DataFrame({
            '年份': [f'第{i}年' for i in range(1, Constants.OPERATION_PERIOD + 1)],
            '税后净营业利润(万元)': nopat_list,
            '资本占用(万元)': capital_list,
            '资本成本(万元)': [c * wacc for c in capital_list],
            f'EVA(万元,WACC={wacc:.0%})': eva_list,
            'EVA累计(万元)': pd.Series(eva_list).cumsum().tolist(),
        })

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"EVA测算表已保存到: {filename}")

        return table

    def export_sensitivity_summary_table(
        self,
        filename: Optional[str] = None,
        factors: Optional[list] = None,
        variation: float = 0.10
    ) -> pd.DataFrame:
        """
        导出敏感性系数和临界点分析表

        Args:
            filename: 输出文件名，如 'sensitivity_summary.csv'
            factors: 要分析的因素列表，默认分析主要因素
            variation: 变化范围，默认±10%

        Returns:
            敏感性汇总表 DataFrame
        """
        if self.df is None:
            raise CalculationError("请先运行 calculate_cash_flow()")

        metrics = self.get_metrics()
        base_irr = metrics['全投资IRR(税前)']

        # 确定要分析的因素
        if factors is None:
            if self.mode == Constants.MODE_SELF_CONSUMPTION:
                factors = ['static_invest', 'hours', 'retail_price', 'feedin_price', 'self_consumption_ratio']
                factor_names = ['静态投资', '利用小时数', '零售电价', '上网电价', '自用比例']
            else:
                factors = ['static_invest', 'hours', 'price_tax_inc']
                factor_names = ['静态投资', '利用小时数', '上网电价']

        results = []

        for factor, name in zip(factors, factor_names):
            # 进行敏感性分析
            sens_df = sensitivity_analysis(self.p, factor, variation_range=variation, steps=3)

            # 提取关键信息
            irr_values = sens_df['IRR(税前)%'].dropna().values
            if len(irr_values) >= 2:
                irr_max = irr_values.max()
                irr_min = irr_values.min()
                irr_range = irr_max - irr_min

                # 计算敏感度系数
                sensitivity_coefficient = irr_range / base_irr / (2 * variation)

                # 计算临界点（IRR=0时的变化率）
                # 简化计算：假设线性关系
                critical_change = -base_irr / (irr_values[2] - irr_values[0]) * variation * 2 if irr_values[0] != irr_values[2] else 0

                results.append({
                    '敏感因素': name,
                    '敏感度系数': round(sensitivity_coefficient, 3),
                    'IRR变化范围(%)': f"{irr_min:.2f} ~ {irr_max:.2f}",
                    'IRR变化幅度(%)': round(irr_range, 2),
                    '临界点(%)': f"{critical_change:.1f}%" if critical_change != 0 else "无法达到",
                    '敏感程度': '高' if abs(sensitivity_coefficient) > 1.5 else '中' if abs(sensitivity_coefficient) > 0.8 else '低',
                })

        table = pd.DataFrame(results)

        if filename:
            table.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"敏感性汇总表已保存到: {filename}")

        return table


# ==============================================================================
# 敏感性分析
# ==============================================================================

def sensitivity_analysis(
    base_params: Dict[str, Any],
    factor: str,
    variation_range: float = 0.1,
    steps: int = 5
) -> pd.DataFrame:
    """
    单因素敏感性分析

    分析某个因素变化对 IRR 的影响

    Args:
        base_params: 基础项目参数
        factor: 要分析的因素，支持:
            - 'static_invest': 静态投资
            - 'price'或'price_tax_inc': 电价
            - 'hours'或'gen_hours': 发电量/利用小时数
            - 'retail_price': 零售电价（自发自用模式）
            - 'feedin_price': 上网电价（自发自用模式）
            - 'self_consumption_ratio': 自用比例（自发自用模式）
        variation_range: 变化范围，默认 ±10%
        steps: 分析步数，默认 5 步（-10%, -5%, 0%, +5%, +10%）

    Returns:
        敏感性分析结果 DataFrame
    """
    results = []

    # 获取基准值
    base_value = base_params.get(factor)
    if base_value is None:
        # 处理参数名映射
        if factor in ['price', 'price_tax_inc']:
            base_value = base_params.get('price_tax_inc')
        elif factor in ['hours', 'gen_hours']:
            base_value = base_params.get('hours', 1000)
        else:
            raise ValueError(f"未知的因素: {factor}")

    # 生成变化序列
    variations = np.linspace(-variation_range, variation_range, steps)

    for var in variations:
        params_temp = base_params.copy()
        new_value = base_value * (1 + var)
        params_temp[factor] = new_value

        # 如果是自发自用模式，需要特殊处理
        if factor == 'retail_price' or factor == 'feedin_price' or factor == 'self_consumption_ratio':
            params_temp[factor] = new_value

        try:
            project = PVProject(params_temp)
            project.calculate_cash_flow()
            metrics = project.get_metrics()
            irr = metrics['全投资IRR(税前)']

            results.append({
                '因素': factor,
                '变化率': f'{var*100:+.1f}%',
                '数值': new_value,
                'IRR(税前)%': irr,
                'IRR变化': f"{irr - metrics['全投资IRR(税前)']:+.2f}" if var == 0 else f""
            })
        except Exception as e:
            logger.error(f"敏感性分析失败 (变化率={var*100:.1f}%): {e}")
            results.append({
                '因素': factor,
                '变化率': f'{var*100:+.1f}%',
                '数值': new_value,
                'IRR(税前)%': None,
                'IRR变化': "计算失败"
            })

    df = pd.DataFrame(results)

    # 计算敏感度系数
    if len(df) > 0 and df['IRR(税前)%'].notna().sum() >= 2:
        base_irr = df.loc[df['变化率'] == '0.0%', 'IRR(税前)%'].values[0] if '0.0%' in df['变化率'].values else df['IRR(税前)%'].values[len(df)//2]
        df['敏感度系数'] = df.apply(lambda row: (
            (row['IRR(税前)%'] - base_irr) / base_irr / (float(row['变化率'].replace('%', '')) / 100)
            if row['IRR(税前)%'] is not None and row['变化率'] != '0.0%' else 0.0
        ), axis=1)

    logger.info(f"敏感性分析完成: 因素={factor}")
    return df


# ==============================================================================
# 高级功能: 反向求解 (Goal Seek)
# ==============================================================================

def goal_seek_investment(
    target_irr: float,
    params: Dict[str, Any],
    min_invest: Optional[float] = None,
    max_invest: Optional[float] = None
) -> Optional[float]:
    """
    给定目标IRR，反推最大允许的静态投资

    使用 Scipy Brentq 算法进行快速求解

    Args:
        target_irr: 目标全投资IRR (税前)，如 8.0 表示 8%
        params: 项目参数字典
        min_invest: 搜索下限 (万元)，默认1000
        max_invest: 搜索上限 (万元)，默认100000

    Returns:
        最大允许静态投资 (万元)，如果求解失败则返回 None
    """
    min_inv = min_invest or Constants.MIN_INVEST
    max_inv = max_invest or Constants.MAX_INVEST

    def objective(invest_guess: float) -> float:
        p_temp = params.copy()
        p_temp['static_invest'] = invest_guess

        if 'deductible_tax' not in p_temp:
            p_temp['deductible_tax'] = (
                invest_guess / (1 + Constants.VAT_RATE) * Constants.VAT_RATE
            )

        project = PVProject(p_temp)
        project.calculate_cash_flow()
        metrics = project.get_metrics()
        return metrics['全投资IRR(税前)'] - target_irr

    try:
        limit_invest = optimize.brentq(objective, min_inv, max_inv)
        logger.info(f"Goal Seek 成功: 目标IRR={target_irr}% -> 最大投资={limit_invest:.2f}万元")
        return limit_invest
    except ValueError as e:
        logger.error(f"Goal Seek 失败: 目标IRR {target_irr}% 在范围[{min_inv}, {max_inv}]内无解")
        return None
    except Exception as e:
        logger.error(f"Goal Seek 失败: {e}")
        return None


# ==============================================================================
# 演示与测试
# ==============================================================================

def demo_qionghai_project() -> None:
    """
    琼海 100MW 集中式光伏项目演示（全额上网模式）

    对标数据 (木联能软件):
        - 建设期利息: 780.18 万元
        - 总投资: 41080.18 万元
        - 全投资IRR(税前): 11.35%
    """
    print("\n" + "=" * 60)
    print("🌟 PyPV-Eval v1.1.0 - 光伏项目技经评价引擎")
    print("=" * 60)

    qionghai_params = {
        'capacity_mw': 100.0,
        'static_invest': 40000.0,
        'capital_ratio': 0.20,
        'loan_rate': 0.04876,
        'hours': 1500,
        'price_tax_inc': 0.40,
        'deductible_tax': 4000.0
    }

    try:
        print("\n📊 正在执行琼海项目 (100MW) 计算...")
        project = PVProject(qionghai_params)
        project.calculate_cash_flow()
        metrics = project.get_metrics()

        print("\n" + "-" * 60)
        print("✅ 琼海项目 (100MW) 技经评价报告")
        print("-" * 60)
        print(f"💰 项目总投资:      {metrics['总投资']:>12} 万元")
        print(f"🏗️  建设期利息:     {metrics['建设期利息']:>12} 万元  (对标: 780.18)")
        print(f"📈 IRR (税前):      {metrics['全投资IRR(税前)']:>12}%       (对标: 11.35%)")
        print(f"📉 IRR (税后):      {metrics['全投资IRR(税后)']:>12}%")
        print(f"📅 投资回收期:      {metrics['投资回收期(年)']:>12} 年")
        print("-" * 60)

        # 反向求解演示
        target = 8.0
        print(f"\n🔮 [决策辅助] 若目标 IRR 为 {target}%:")
        limit = goal_seek_investment(target, qionghai_params)
        if limit is not None:
            print(f"👉 最大允许静态投资:  {limit:>10.2f} 万元")
            print(f"👉 相比当前方案盈余:  {limit - 40000:>10.2f} 万元")
        print("=" * 60)

    except (InputValidationError, CalculationError) as e:
        print(f"\n❌ 错误: {e}")
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")


def demo_self_consumption_project() -> None:
    """
    工商业分布式光伏项目演示（自发自用模式）

    典型场景：
        - 1MW 工商业屋顶光伏
        - 自用比例 80%
        - 工商业电价 0.8 元/kWh
        - 余电上网电价 0.4 元/kWh
    """
    print("\n" + "=" * 60)
    print("🏢 工商业分布式光伏项目演示（自发自用模式）")
    print("=" * 60)

    distributed_params = {
        'capacity_mw': 1.0,              # 1MW
        'static_invest': 350.0,           # 350万元（约3.5元/W）
        'mode': 'self_consumption',
        'self_consumption_ratio': 0.8,    # 80%自用
        'retail_price': 0.85,             # 工商业电价 0.85元/kWh
        'feedin_price': 0.42,             # 余电上网价 0.42元/kWh
        'hours': 1100,                    # 年利用小时数
        'capital_ratio': 0.3,
        'loan_rate': 0.04,
    }

    try:
        print("\n📊 项目参数:")
        print(f"   装机容量: {distributed_params['capacity_mw']} MW")
        print(f"   静态投资: {distributed_params['static_invest']} 万元")
        print(f"   自用比例: {distributed_params['self_consumption_ratio']:.0%}")
        print(f"   零售电价: {distributed_params['retail_price']} 元/kWh")
        print(f"   上网电价: {distributed_params['feedin_price']} 元/kWh")

        project = PVProject(distributed_params)
        project.calculate_cash_flow()
        metrics = project.get_metrics()

        print("\n" + "-" * 60)
        print("✅ 工商业分布式项目技经评价报告")
        print("-" * 60)
        print(f"💰 项目总投资:      {metrics['总投资']:>12} 万元")
        print(f"🏗️  建设期利息:     {metrics['建设期利息']:>12} 万元")
        print(f"📈 IRR (税前):      {metrics['全投资IRR(税前)']:>12}%")
        print(f"📉 IRR (税后):      {metrics['全投资IRR(税后)']:>12}%")
        print(f"📅 投资回收期:      {metrics['投资回收期(年)']:>12} 年")
        print("-" * 60)

        # 反向求解演示
        target = 12.0  # 分布式项目目标IRR通常较高
        print(f"\n🔮 [决策辅助] 若目标 IRR 为 {target}%:")
        limit = goal_seek_investment(target, distributed_params)
        if limit is not None:
            print(f"👉 最大允许静态投资:  {limit:>10.2f} 万元")
            print(f"👉 相比当前方案盈余:  {limit - distributed_params['static_invest']:>10.2f} 万元")
        print("=" * 60)

    except (InputValidationError, CalculationError) as e:
        print(f"\n❌ 错误: {e}")
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")


if __name__ == "__main__":
    # 运行两个演示
    demo_qionghai_project()
    demo_self_consumption_project()
