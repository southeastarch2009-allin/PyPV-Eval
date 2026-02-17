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
