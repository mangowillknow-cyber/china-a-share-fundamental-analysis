# -*- coding: utf-8 -*-
"""
================================================================================
A股基本面分析作品集 - 模块一：数据获取与清洗工程
================================================================================
目标：展示 SQL/Pandas 数据清洗能力
内容：
  1. 模拟生成沪深A股400家公司2019-2023年三张财务报表数据
  2. 注入真实世界的数据质量问题
  3. 执行6步清洗流水线
  4. 输出数据质量报告

作者：数据分析求职作品集
日期：2024
"""

import numpy as np
import pandas as pd
import sys
import os
import io
import warnings
warnings.filterwarnings('ignore')

# 修复Windows控制台中文输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ======================== 全局配置 ========================
SEED = 2024                          # 全局随机种子，确保可复现
YEARS = [2019, 2020, 2021, 2022, 2023]  # 分析时间跨度

# 行业配置：名称、代码前缀、公司数量、财务参数范围
INDUSTRY_CONFIG = {
    '制造业': {
        'prefix': '6000', 'n': 80,
        'revenue_logmean': 3.0, 'revenue_logstd': 1.0,   # 营收对数正态参数（亿元）
        'net_margin': (0.03, 0.12),                        # 净利率范围
        'asset_turnover': (0.4, 0.8),                      # 资产周转率范围
        'debt_ratio': (0.40, 0.65),                        # 资产负债率范围
        'rd_intensity': (0.03, 0.08),                      # 研发强度范围
    },
    '信息技术': {
        'prefix': '3000', 'n': 80,
        'revenue_logmean': 2.0, 'revenue_logstd': 1.2,
        'net_margin': (0.08, 0.25),
        'asset_turnover': (0.3, 0.7),
        'debt_ratio': (0.20, 0.45),
        'rd_intensity': (0.10, 0.25),
    },
    '金融业': {
        'prefix': '6013', 'n': 80,
        'revenue_logmean': 4.0, 'revenue_logstd': 1.5,
        'net_margin': (0.15, 0.35),
        'asset_turnover': (0.02, 0.06),
        'debt_ratio': (0.85, 0.95),
        'rd_intensity': (0.001, 0.01),  # 金融业研发强度极低
    },
    '消费品': {
        'prefix': '0008', 'n': 80,
        'revenue_logmean': 2.5, 'revenue_logstd': 1.0,
        'net_margin': (0.05, 0.18),
        'asset_turnover': (0.6, 1.2),
        'debt_ratio': (0.30, 0.55),
        'rd_intensity': (0.01, 0.04),
    },
    '能源': {
        'prefix': '6005', 'n': 80,
        'revenue_logmean': 3.5, 'revenue_logstd': 1.2,
        'net_margin': (0.02, 0.10),
        'asset_turnover': (0.3, 0.7),
        'debt_ratio': (0.45, 0.75),
        'rd_intensity': (0.01, 0.03),
    },
}


# ======================== 第一部分：数据生成 ========================
print("=" * 70)
print("第一部分：模拟生成A股上市公司财务数据")
print("=" * 70)

def generate_company_universe(rng):
    """生成400家公司的基础信息表

    参数:
        rng: numpy随机数生成器

    返回:
        pd.DataFrame: 包含股票代码、公司名称、行业的基础信息表
    """
    companies = []  # 存储所有公司信息的列表

    for industry_name, config in INDUSTRY_CONFIG.items():
        for i in range(config['n']):
            # 生成股票代码：行业前缀 + 4位序号
            stock_code = f"{config['prefix']}{i:04d}"
            # 生成公司名称：行业 + 序号
            company_name = f"{industry_name}第{i+1:03d}号公司"
            companies.append({
                '股票代码': stock_code,
                '公司名称': company_name,
                '行业': industry_name,
            })

    df = pd.DataFrame(companies)  # 转为DataFrame
    print(f"  已生成 {len(df)} 家公司的基础信息")
    print(f"  行业分布：\n{df['行业'].value_counts().to_string()}")
    return df


def generate_financial_data(companies_df, rng):
    """基于公司基础信息，生成三张财务报表的完整数据

    参数:
        companies_df: 公司基础信息表
        rng: numpy随机数生成器

    返回:
        tuple: (利润表DataFrame, 资产负债表DataFrame, 现金流量表DataFrame)
    """
    income_records = []      # 利润表记录列表
    balance_records = []     # 资产负债表记录列表
    cashflow_records = []    # 现金流量表记录列表

    for _, company in companies_df.iterrows():
        stock_code = company['股票代码']
        company_name = company['公司名称']
        industry = company['行业']
        config = INDUSTRY_CONFIG[industry]

        # 为每家公司生成一个稳定的"基础营收"（对数正态分布，模拟真实市场右偏特征）
        base_revenue = rng.lognormal(mean=config['revenue_logmean'], sigma=config['revenue_logstd'])

        # 为每家公司确定一个稳定的净利率水平
        base_net_margin = rng.uniform(*config['net_margin'])
        # 确定稳定的资产周转率
        base_turnover = rng.uniform(*config['asset_turnover'])
        # 确定稳定的资产负债率
        base_debt_ratio = rng.uniform(*config['debt_ratio'])
        # 确定稳定的研发强度
        base_rd_intensity = rng.uniform(*config['rd_intensity'])

        for year in YEARS:
            # --- 添加年度随机波动（±5%），模拟真实财务数据的年际变化 ---
            annual_shock = rng.normal(0, 0.05)  # 年度冲击因子

            # ===== 利润表生成 =====
            # 营业收入 = 基础营收 × 年度增长趋势 × 随机波动
            year_growth = 1 + (year - 2019) * 0.03  # 年均3%增长趋势
            revenue = base_revenue * year_growth * (1 + annual_shock)
            revenue = max(revenue, 0.1)  # 确保营收不为负

            # 营业成本 = 营业收入 × (1 - 毛利率)
            gross_margin = rng.uniform(0.15, 0.45)  # 毛利率
            cost_of_revenue = revenue * (1 - gross_margin)

            # 期间费用（销售/管理/研发/财务），均以营收的一定比例计算
            selling_expense = revenue * rng.uniform(0.02, 0.08)    # 销售费用
            admin_expense = revenue * rng.uniform(0.03, 0.10)      # 管理费用
            rd_expense = revenue * base_rd_intensity * (1 + rng.normal(0, 0.1))  # 研发费用
            finance_expense = revenue * rng.uniform(0.005, 0.03)   # 财务费用

            # 资产减值损失（可为负，表示转回）
            impairment_loss = revenue * rng.uniform(-0.01, 0.02)

            # 营业利润 = 营收 - 成本 - 四费 - 减值
            operating_profit = (revenue - cost_of_revenue - selling_expense
                                - admin_expense - rd_expense - finance_expense
                                - impairment_loss)

            # 利润总额 = 营业利润 + 营业外收支（简化处理）
            total_profit = operating_profit * rng.uniform(0.95, 1.05)

            # 所得税费用 = 利润总额 × 税率（约25%，金融业可能不同）
            tax_rate = 0.25 if industry != '金融业' else 0.25
            income_tax = max(total_profit * tax_rate * rng.uniform(0.9, 1.1), 0)

            # 净利润 = 利润总额 - 所得税
            net_income = total_profit - income_tax

            # 归母净利润（约80%-100%的净利润）
            net_income_parent = net_income * rng.uniform(0.80, 1.00)

            income_records.append({
                '股票代码': stock_code,
                '公司名称': company_name,
                '行业': industry,
                '会计年度': year,
                '营业收入': round(revenue, 2),
                '营业成本': round(cost_of_revenue, 2),
                '销售费用': round(selling_expense, 2),
                '管理费用': round(admin_expense, 2),
                '研发费用': round(max(rd_expense, 0), 2),
                '财务费用': round(finance_expense, 2),
                '资产减值损失': round(impairment_loss, 2),
                '营业利润': round(operating_profit, 2),
                '利润总额': round(total_profit, 2),
                '所得税费用': round(income_tax, 2),
                '净利润': round(net_income, 2),
                '归母净利润': round(net_income_parent, 2),
            })

            # ===== 资产负债表生成 =====
            # 总资产 = 营业收入 / 资产周转率（反推）
            total_assets = revenue / max(base_turnover * (1 + rng.normal(0, 0.05)), 0.01)

            # 流动资产各组成部分（按总资产的比例分配）
            cash = total_assets * rng.uniform(0.08, 0.25)           # 货币资金
            receivables = revenue * rng.uniform(0.05, 0.20)         # 应收账款（与营收挂钩）
            inventory = cost_of_revenue * rng.uniform(0.10, 0.30)   # 存货（与成本挂钩）
            current_assets = cash + receivables + inventory + total_assets * rng.uniform(0.02, 0.10)

            # 非流动资产
            fixed_assets = total_assets * rng.uniform(0.15, 0.45)   # 固定资产
            intangible_assets = total_assets * rng.uniform(0.02, 0.15)  # 无形资产
            non_current_assets = fixed_assets + intangible_assets + total_assets * rng.uniform(0.02, 0.10)

            # 校验总资产 = 流动资产 + 非流动资产
            total_assets_actual = current_assets + non_current_assets

            # 负债端（按负债率分配）
            short_term_debt = total_assets_actual * rng.uniform(0.05, 0.20)   # 短期借款
            payables = revenue * rng.uniform(0.05, 0.15)                      # 应付账款
            current_liabilities = short_term_debt + payables + total_assets_actual * rng.uniform(0.05, 0.15)

            long_term_debt = total_assets_actual * rng.uniform(0.05, 0.25)    # 长期借款
            non_current_liabilities = long_term_debt + total_assets_actual * rng.uniform(0.02, 0.08)

            total_liabilities = current_liabilities + non_current_liabilities
            # 确保负债率在行业合理范围内
            target_debt = total_assets_actual * base_debt_ratio
            scale_factor = target_debt / max(total_liabilities, 1)
            total_liabilities = total_liabilities * scale_factor
            current_liabilities = current_liabilities * scale_factor
            non_current_liabilities = non_current_liabilities * scale_factor

            owners_equity = total_assets_actual - total_liabilities  # 所有者权益
            equity_parent = owners_equity * rng.uniform(0.85, 1.00)  # 归母权益

            balance_records.append({
                '股票代码': stock_code,
                '公司名称': company_name,
                '行业': industry,
                '会计年度': year,
                '货币资金': round(cash, 2),
                '应收账款': round(receivables, 2),
                '存货': round(inventory, 2),
                '流动资产合计': round(current_assets, 2),
                '固定资产': round(fixed_assets, 2),
                '无形资产': round(intangible_assets, 2),
                '非流动资产合计': round(non_current_assets, 2),
                '资产总计': round(total_assets_actual, 2),
                '短期借款': round(short_term_debt, 2),
                '应付账款': round(payables, 2),
                '流动负债合计': round(current_liabilities, 2),
                '长期借款': round(long_term_debt, 2),
                '非流动负债合计': round(non_current_liabilities, 2),
                '负债合计': round(total_liabilities, 2),
                '所有者权益合计': round(owners_equity, 2),
                '归母股东权益': round(equity_parent, 2),
            })

            # ===== 现金流量表生成 =====
            # 经营现金流 ≈ 净利润 × (0.8~1.3) + 随机波动
            operating_cf = net_income * rng.uniform(0.8, 1.3) + rng.normal(0, revenue * 0.02)
            # 投资现金流通常为负（资本支出）
            investing_cf = -total_assets_actual * rng.uniform(0.02, 0.08) + rng.normal(0, revenue * 0.01)
            # 筹资现金流可正可负
            financing_cf = rng.normal(0, revenue * 0.05)
            # 现金净增加额 = 三项之和
            cash_net_change = operating_cf + investing_cf + financing_cf

            cashflow_records.append({
                '股票代码': stock_code,
                '公司名称': company_name,
                '行业': industry,
                '会计年度': year,
                '经营活动现金流净额': round(operating_cf, 2),
                '投资活动现金流净额': round(investing_cf, 2),
                '筹资活动现金流净额': round(financing_cf, 2),
                '现金及等价物净增加额': round(cash_net_change, 2),
            })

    # 将记录列表转为DataFrame
    df_income = pd.DataFrame(income_records)
    df_balance = pd.DataFrame(balance_records)
    df_cashflow = pd.DataFrame(cashflow_records)

    return df_income, df_balance, df_cashflow


def inject_quality_issues(df_income, df_balance, df_cashflow, rng):
    """向原始数据注入真实世界中常见的数据质量问题

    注入问题类型：
    1. 随机缺失值（模拟报表填报遗漏）
    2. 重复记录（模拟数据录入重复）
    3. 极端异常值（模拟计量单位错误）
    4. 行业标签不一致（模拟不同数据源的命名差异）
    5. 类型污染（模拟字符串混入数值列）

    参数:
        df_income, df_balance, df_cashflow: 三张原始财务报表
        rng: 随机数生成器

    返回:
        tuple: 注入问题后的三张表
    """
    print("\n--- 注入数据质量问题 ---")

    # 为三张表分别注入问题，使用独立副本避免修改原始数据
    income = df_income.copy()
    balance = df_balance.copy()
    cashflow = df_cashflow.copy()

    n_income = len(income)
    n_balance = len(balance)
    n_cashflow = len(cashflow)

    # 问题1：随机缺失值（3%的数值单元格变为NaN）
    numeric_cols_income = ['营业收入', '营业成本', '销售费用', '管理费用', '研发费用',
                           '财务费用', '资产减值损失', '营业利润', '利润总额',
                           '所得税费用', '净利润', '归母净利润']
    for col in numeric_cols_income:
        mask = rng.random(n_income) < 0.03  # 3%的概率缺失
        income.loc[mask, col] = np.nan

    numeric_cols_balance = ['货币资金', '应收账款', '存货', '流动资产合计',
                            '固定资产', '无形资产', '非流动资产合计', '资产总计',
                            '短期借款', '应付账款', '流动负债合计', '长期借款',
                            '非流动负债合计', '负债合计', '所有者权益合计', '归母股东权益']
    for col in numeric_cols_balance:
        mask = rng.random(n_balance) < 0.03
        balance.loc[mask, col] = np.nan

    numeric_cols_cashflow = ['经营活动现金流净额', '投资活动现金流净额',
                              '筹资活动现金流净额', '现金及等价物净增加额']
    for col in numeric_cols_cashflow:
        mask = rng.random(n_cashflow) < 0.03
        cashflow.loc[mask, col] = np.nan

    # 特殊处理：金融业公司不披露研发费用，设置为缺失
    finance_mask = income['行业'] == '金融业'
    income.loc[finance_mask, '研发费用'] = np.nan
    print(f"  [缺失值] 已为3%的数值单元格注入NaN，金融业研发费用全部设为缺失")

    # 问题2：重复记录（2%的行复制一份追加到表尾）
    dup_count = int(n_income * 0.02)
    dup_idx = rng.choice(n_income, size=dup_count, replace=False)
    income = pd.concat([income, income.iloc[dup_idx]], ignore_index=True)
    print(f"  [重复行] 利润表注入 {dup_count} 行重复记录（{n_income} -> {len(income)} 行）")

    dup_count_b = int(n_balance * 0.02)
    dup_idx_b = rng.choice(n_balance, size=dup_count_b, replace=False)
    balance = pd.concat([balance, balance.iloc[dup_idx_b]], ignore_index=True)

    dup_count_c = int(n_cashflow * 0.02)
    dup_idx_c = rng.choice(n_cashflow, size=dup_count_c, replace=False)
    cashflow = pd.concat([cashflow, cashflow.iloc[dup_idx_c]], ignore_index=True)

    # 问题3：极端异常值（1%的营业收入放大10倍，模拟计量单位错误）
    outlier_count = int(n_income * 0.01)
    outlier_idx = rng.choice(n_income, size=outlier_count, replace=False)
    income.loc[outlier_idx, '营业收入'] = income.loc[outlier_idx, '营业收入'] * 10
    print(f"  [异常值] 已将 {outlier_count} 条营业收入放大10倍（模拟单位错误）")

    # 问题4：行业标签不一致（约3%的记录使用变体名称）
    industry_variants = {
        '制造业': ['制造', '制造业', '制造行业'],
        '信息技术': ['IT', '信息技术', '信息产业'],
        '金融业': ['金融', '金融业', '金融行业'],
        '消费品': ['消费品', '消费', '消费品行业'],
        '能源': ['能源', '能源业', '能源行业'],
    }
    # 为部分记录使用变体标签
    for df in [income, balance, cashflow]:
        for original, variants in industry_variants.items():
            original_mask = df['行业'] == original
            original_indices = df[original_mask].index.tolist()
            # 选5%的记录改为变体
            n_change = max(1, int(len(original_indices) * 0.05))
            change_idx = rng.choice(original_indices, size=min(n_change, len(original_indices)), replace=False)
            variant = variants[rng.integers(0, len(variants))]  # 随机选一个变体
            df.loc[change_idx, '行业'] = variant
    print(f"  [标签不一致] 已为约5%的行业标签注入变体名称")

    # 问题5：类型污染（将部分数值转为带空格的字符串）
    # 只污染财务金额列，排除年份等分类变量
    financial_numeric_cols = ['营业收入', '营业成本', '销售费用', '货币资金', '应收账款']
    for df in [income, balance, cashflow]:
        for col in financial_numeric_cols:
            if col in df.columns:
                n_pollute = int(len(df) * 0.03)
                pollute_idx = rng.choice(len(df), size=n_pollute, replace=False)
                # 先将列转为object类型，再插入字符串值（pandas 3.0 CopyOnWrite 要求类型兼容）
                df[col] = df[col].astype(object)
                df.loc[pollute_idx, col] = df.loc[pollute_idx, col].apply(
                    lambda x: f" {x} " if pd.notna(x) else x
                )
    print(f"  [类型污染] 已将约3%的财务数值转为带空格的字符串")

    # 打乱行顺序（模拟数据采集时的无序状态）
    income = income.sample(frac=1, random_state=42).reset_index(drop=True)
    balance = balance.sample(frac=1, random_state=42).reset_index(drop=True)
    cashflow = cashflow.sample(frac=1, random_state=42).reset_index(drop=True)

    return income, balance, cashflow


# ======================== 第二部分：数据清洗流水线 ========================
print("\n" + "=" * 70)
print("第二部分：数据清洗流水线（6步）")
print("=" * 70)


def step1_merge_tables(df_income, df_balance, df_cashflow):
    """第1步：将三张表按【股票代码+会计年度】进行全外连接合并

    目的：生成一张包含所有财务信息的宽表，便于后续分析
    注意：使用 outer join 确保不丢失任何一方的数据
    """
    print("\n" + "-" * 50)
    print("【第1步：三表全外连接合并】")
    print("-" * 50)

    # 打印合并前各表的形状
    print(f"  利润表：{df_income.shape[0]} 行 × {df_income.shape[1]} 列")
    print(f"  资产负债表：{df_balance.shape[0]} 行 × {df_balance.shape[1]} 列")
    print(f"  现金流量表：{df_cashflow.shape[0]} 行 × {df_cashflow.shape[1]} 列")

    # 定义合并键
    merge_keys = ['股票代码', '会计年度']

    # 第一次合并：利润表 + 资产负债表
    df_wide = pd.merge(df_income, df_balance, on=merge_keys, how='outer', suffixes=('', '_bs'))
    print(f"  利润表 ∪ 资产负债表：{df_wide.shape[0]} 行 × {df_wide.shape[1]} 列")

    # 第二次合并：上一步结果 + 现金流量表
    df_wide = pd.merge(df_wide, df_cashflow, on=merge_keys, how='outer', suffixes=('', '_cf'))
    print(f"  最终宽表：{df_wide.shape[0]} 行 × {df_wide.shape[1]} 列")

    # 处理合并后可能产生的重复列名（如行业、公司名称在三张表中都存在）
    # 保留第一张表的行业和公司名称，删除重复列
    for col in df_wide.columns:
        if col.endswith('_bs') or col.endswith('_cf'):
            base_col = col.replace('_bs', '').replace('_cf', '')
            if base_col in df_wide.columns:
                # 用后表的非空值填充前表的空值
                df_wide[base_col] = df_wide[base_col].fillna(df_wide[col])
                df_wide = df_wide.drop(columns=[col])

    # 确保关键列存在
    if '行业' not in df_wide.columns and '行业_bs' in df_wide.columns:
        df_wide['行业'] = df_wide['行业_bs']
    if '公司名称' not in df_wide.columns and '公司名称_bs' in df_wide.columns:
        df_wide['公司名称'] = df_wide['公司名称_bs']

    # 清理残余的 _bs/_cf 后缀列
    drop_cols = [c for c in df_wide.columns if c.endswith('_bs') or c.endswith('_cf')]
    if drop_cols:
        for col in drop_cols:
            base = col.replace('_bs', '').replace('_cf', '')
            if base not in df_wide.columns:
                df_wide = df_wide.rename(columns={col: base})
            else:
                df_wide = df_wide.drop(columns=[col])

    print(f"  合并后宽表形状：{df_wide.shape[0]} 行 × {df_wide.shape[1]} 列")
    return df_wide


def step2_deduplicate(df):
    """第2步：识别并处理重复记录

    策略：同一【股票代码+会计年度】出现多次时，保留第一条（模拟保留最新填报）
    """
    print("\n" + "-" * 50)
    print("【第2步：去重处理】")
    print("-" * 50)

    n_before = len(df)
    # 统计重复行数
    dup_mask = df.duplicated(subset=['股票代码', '会计年度'], keep='first')
    n_dup = dup_mask.sum()
    print(f"  去重前总行数：{n_before}")
    print(f"  按【股票代码+会计年度】发现重复记录：{n_dup} 行")

    # 删除重复行，保留每组的第一条
    df = df.drop_duplicates(subset=['股票代码', '会计年度'], keep='first').reset_index(drop=True)
    print(f"  去重后总行数：{len(df)}（删除 {n_before - len(df)} 行）")

    return df


def step3_standardize_industry(df):
    """第3步：标准化行业标签

    将各种变体名称统一映射回标准名称
    """
    print("\n" + "-" * 50)
    print("【第3步：行业标签标准化】")
    print("-" * 50)

    # 打印标准化前的行业分布
    print("  标准化前行业分布：")
    for name, count in df['行业'].value_counts().items():
        print(f"    {name}: {count}")

    # 定义变体到标准名称的映射字典
    industry_mapping = {
        '制造': '制造业',
        '制造行业': '制造业',
        'IT': '信息技术',
        '信息产业': '信息技术',
        '金融': '金融业',
        '金融行业': '金融业',
        '消费': '消费品',
        '消费品行业': '消费品',
        '能源': '能源',
        '能源业': '能源',
        '能源行业': '能源',
    }

    # 执行替换
    df['行业'] = df['行业'].replace(industry_mapping)

    # 打印标准化后的行业分布
    print("\n  标准化后行业分布：")
    for name, count in df['行业'].value_counts().items():
        print(f"    {name}: {count}")

    return df


def step4_type_coercion(df):
    """第4步：类型转换——将被污染的字符串数值恢复为数值类型

    处理步骤：去除前后空格 → 尝试转为数值 → 无法转换的设为NaN
    """
    print("\n" + "-" * 50)
    print("【第4步：类型转换】")
    print("-" * 50)

    # 获取所有非标识列（排除代码、名称、行业、年份）
    id_cols = ['股票代码', '公司名称', '行业', '会计年度']
    numeric_cols = [c for c in df.columns if c not in id_cols]

    coercion_report = {}  # 记录每列的转换情况

    for col in numeric_cols:
        # 先统计原始缺失数
        original_null = df[col].isna().sum()

        # 去除字符串前后空格
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            # 将空字符串转为NaN
            df.loc[df[col] == '', col] = np.nan
            df.loc[df[col] == 'nan', col] = np.nan

        # 尝试转为数值类型（无法转换的自动变为NaN）
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # 统计转换后新增的NaN数量
        new_null = df[col].isna().sum()
        added_nulls = new_null - original_null
        if added_nulls > 0:
            coercion_report[col] = added_nulls

    # 打印转换报告
    if coercion_report:
        print("  类型转换产生的新增NaN：")
        for col, count in coercion_report.items():
            print(f"    {col}: +{count} 个NaN")
    else:
        print("  未发现类型污染")

    return df


def step5_impute_missing(df):
    """第5步：缺失值填补（4层策略）

    策略说明：
    第1层 - 领域规则：金融业研发费用缺失填0（金融业不单独披露研发费用）
    第2层 - 同行业同年度中位数填充：适用于营业收入、总资产等关键变量
    第3层 - 时间序列前向填充：适用于资产负债表的余额类项目（按公司分组向前填充）
    第4层 - 删除超过50%字段缺失的行（极端情况）
    """
    print("\n" + "-" * 50)
    print("【第5步：缺失值填补（4层策略）】")
    print("-" * 50)

    # 统计填补前各列缺失率
    before_missing = df.isnull().sum()
    before_pct = (before_missing / len(df) * 100).round(2)
    print("  填补前缺失率（%）：")
    for col in df.columns:
        if before_missing[col] > 0:
            print(f"    {col}: {before_pct[col]}%（{before_missing[col]}条）")

    # --- 第1层：领域规则填补 ---
    # 金融业的研发费用缺失是合理的（该行业通常不单独披露），填0
    finance_mask = df['行业'] == '金融业'
    n_finance_rd_null = (finance_mask & df['研发费用'].isna()).sum()
    df.loc[finance_mask & df['研发费用'].isna(), '研发费用'] = 0
    print(f"\n  [第1层] 金融业研发费用缺失 → 填0（填补 {n_finance_rd_null} 条）")

    # --- 第2层：同行业同年度中位数填充 ---
    # 适用于营收、利润等横截面变量
    cross_section_cols = ['营业收入', '营业成本', '销售费用', '管理费用', '财务费用',
                          '资产减值损失', '营业利润', '利润总额', '所得税费用',
                          '净利润', '归母净利润']
    filled_count_2 = 0
    for col in cross_section_cols:
        if col in df.columns:
            # 按行业和年度分组，用每组中位数填充该组内的缺失值
            median_vals = df.groupby(['行业', '会计年度'])[col].transform('median')
            mask = df[col].isna()
            df.loc[mask, col] = median_vals[mask]
            filled_count_2 += mask.sum()
    print(f"  [第2层] 同行业同年度中位数填充（填补 {filled_count_2} 个单元格）")

    # --- 第3层：时间序列前向填充 ---
    # 适用于资产负债表的余额类项目（同一公司不同年份间变化较平缓）
    balance_cols = ['货币资金', '应收账款', '存货', '流动资产合计',
                    '固定资产', '无形资产', '非流动资产合计', '资产总计',
                    '短期借款', '应付账款', '流动负债合计', '长期借款',
                    '非流动负债合计', '负债合计', '所有者权益合计', '归母股东权益']
    filled_count_3 = 0
    for col in balance_cols:
        if col in df.columns:
            # 先按公司分组排序，再前向填充
            df = df.sort_values(['股票代码', '会计年度'])
            mask_before = df[col].isna()
            df[col] = df.groupby('股票代码')[col].ffill()  # 前向填充
            # 后向填充处理第一年可能的缺失
            df[col] = df.groupby('股票代码')[col].bfill()
            filled_count_3 += (mask_before & df[col].notna()).sum()
    print(f"  [第3层] 时间序列前向+后向填充（填补 {filled_count_3} 个单元格）")

    # --- 第4层：删除极端缺失行（超过50%字段缺失） ---
    # 计算每行的缺失比例
    col_count = len(df.columns) - 4  # 排除标识列
    missing_per_row = df.drop(columns=['股票代码', '公司名称', '行业', '会计年度']).isna().sum(axis=1)
    extreme_missing = missing_per_row > col_count * 0.5
    n_extreme = extreme_missing.sum()
    df = df[~extreme_missing].reset_index(drop=True)
    print(f"  [第4层] 删除超过50%字段缺失的行（删除 {n_extreme} 行）")

    # 统计填补后的缺失情况
    after_missing = df.isnull().sum()
    remaining = after_missing[after_missing > 0]
    if len(remaining) > 0:
        print(f"\n  填补后仍存在缺失的列：")
        for col, count in remaining.items():
            print(f"    {col}: {count}（{count/len(df)*100:.2f}%）")
    else:
        print(f"\n  所有缺失值已填补完毕！")

    return df


def step6_winsorize(df):
    """第6步：缩尾处理（Winsorization）

    对所有连续变量在1%和99%分位进行截断
    目的：消除极端异常值对统计分析的影响，同时保留数据的排序信息
    """
    print("\n" + "-" * 50)
    print("【第6步：缩尾处理（1%/99%分位）】")
    print("-" * 50)

    # 选取需要缩尾的数值列（排除年份等分类变量）
    id_cols = ['股票代码', '公司名称', '行业', '会计年度']
    numeric_cols = [c for c in df.columns if c not in id_cols and df[c].dtype in ['float64', 'int64', 'float32']]

    winsorize_report = {}  # 记录每列的缩尾情况

    for col in numeric_cols:
        # 计算1%和99%分位数
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)

        # 统计需要截断的值的数量
        below = (df[col] < q01).sum()
        above = (df[col] > q99).sum()

        if below > 0 or above > 0:
            # 使用clip进行缩尾处理
            df[col] = df[col].clip(lower=q01, upper=q99)
            winsorize_report[col] = {'下界截断': below, '上界截断': above, '下界值': q01, '上界值': q99}

    # 打印缩尾报告
    print(f"  共对 {len(winsorize_report)} 个变量进行了缩尾处理：")
    for col, info in winsorize_report.items():
        print(f"    {col}: 下界截断 {info['下界截断']} 条，上界截断 {info['上界截断']} 条")
        print(f"           范围：[{info['下界值']:.2f}, {info['上界值']:.2f}]")

    return df


def generate_data_quality_report(df_original, df_cleaned, winsorize_info):
    """生成数据质量报告，汇总清洗前后的关键指标

    参数:
        df_original: 清洗前的宽表
        df_cleaned: 清洗后的宽表
        winsorize_info: 缩尾处理的详细信息

    返回:
        dict: 包含所有质量指标的字典
    """
    print("\n" + "=" * 70)
    print("第三部分：数据质量报告")
    print("=" * 70)

    report = {}

    # 样本概览
    report['总样本量（清洗前）'] = len(df_original)
    report['总样本量（清洗后）'] = len(df_cleaned)
    report['字段数量'] = len(df_cleaned.columns)
    report['时间跨度'] = f"{df_cleaned['会计年度'].min()}-{df_cleaned['会计年度'].max()}"
    report['公司数量'] = df_cleaned['股票代码'].nunique()
    report['行业数量'] = df_cleaned['行业'].nunique()

    print(f"\n  【样本概览】")
    print(f"  总样本量（清洗前）：{report['总样本量（清洗前）']}")
    print(f"  总样本量（清洗后）：{report['总样本量（清洗后）']}")
    print(f"  字段数量：{report['字段数量']}")
    print(f"  时间跨度：{report['时间跨度']}")
    print(f"  公司数量：{report['公司数量']}")
    print(f"  行业数量：{report['行业数量']}")

    # 行业分布
    print(f"\n  【行业分布】")
    industry_dist = df_cleaned['行业'].value_counts()
    for ind, count in industry_dist.items():
        pct = count / len(df_cleaned) * 100
        print(f"  {ind}: {count} 条（{pct:.1f}%）")

    # 最终缺失率
    print(f"\n  【最终缺失率】")
    final_missing = df_cleaned.isnull().sum()
    has_missing = False
    for col in df_cleaned.columns:
        pct = final_missing[col] / len(df_cleaned) * 100
        if pct > 0:
            print(f"  {col}: {pct:.2f}%")
            has_missing = True
    if not has_missing:
        print(f"  所有字段缺失率为 0.00% — 数据已完全填补")

    # 输出清洗后的数据预览
    print(f"\n  【清洗后数据前20行预览】")
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 200)
    print(df_cleaned.head(20).to_string())

    return report


# ======================== 主函数 ========================
def main():
    """主函数：串联完整的数据生成与清洗流程"""

    # 初始化随机数生成器
    rng = np.random.default_rng(seed=SEED)

    # 第一部分：生成原始数据
    print("\n>>> 正在生成400家A股上市公司5年财务数据...")
    companies = generate_company_universe(rng)
    df_income, df_balance, df_cashflow = generate_financial_data(companies, rng)

    # 打印原始数据概况
    print(f"\n  原始利润表：{df_income.shape}")
    print(f"  原始资产负债表：{df_balance.shape}")
    print(f"  原始现金流量表：{df_cashflow.shape}")

    # 注入数据质量问题
    print("\n>>> 正在注入数据质量问题...")
    income_dirty, balance_dirty, cashflow_dirty = inject_quality_issues(
        df_income, df_balance, df_cashflow, rng
    )

    # 保留一份原始（脏数据）的宽表副本，用于后续对比
    df_wide_original = step1_merge_tables(income_dirty, balance_dirty, cashflow_dirty)

    # 第二部分：执行清洗流水线
    # 第1步：三表合并
    print("\n>>> 正在执行清洗流水线...")
    df = step1_merge_tables(income_dirty, balance_dirty, cashflow_dirty)

    # 第2步：去重
    df = step2_deduplicate(df)

    # 第3步：行业标签标准化
    df = step3_standardize_industry(df)

    # 第4步：类型转换
    df = step4_type_coercion(df)

    # 第5步：缺失值填补
    df = step5_impute_missing(df)

    # 第6步：缩尾处理
    df = step6_winsorize(df)

    # 第7步：修复会计恒等式（缩尾会破坏 资产 = 负债 + 权益）
    if '资产总计' in df.columns and '负债合计' in df.columns and '所有者权益合计' in df.columns:
        df['所有者权益合计'] = df['资产总计'] - df['负债合计']
        print("\n【第7步：修复会计恒等式】")
        print(f"  已重算 所有者权益合计 = 资产总计 - 负债合计，最大偏差已修正为0")

    # 第三部分：生成数据质量报告
    report = generate_data_quality_report(df_wide_original, df, None)

    # 保存清洗后的数据到CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_financial_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n  清洗后数据已保存至：{output_path}")
    print(f"  最终数据形状：{df.shape[0]} 行 × {df.shape[1]} 列")

    # 保存数据质量报告为CSV
    report_df = pd.DataFrame([report])
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_quality_summary.csv')
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    print(f"  数据质量报告已保存至：{report_path}")

    return df, report


if __name__ == '__main__':
    df_cleaned, quality_report = main()
    print("\n" + "=" * 70)
    print("模块一执行完毕！")
    print("=" * 70)
