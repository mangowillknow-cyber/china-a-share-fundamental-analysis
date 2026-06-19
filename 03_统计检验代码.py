# -*- coding: utf-8 -*-
"""
03_统计检验代码.py
=================
AB测试思维：研发投入对ROE的影响检验
方法：PSM + Welch's T检验 + Cohen's d效应量 + 稳健性检验
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ======================== 数据生成（与模块一/二一致） ========================

def generate_all_data():
    """生成模拟数据（seed=2024，确保一致性）"""
    rng = np.random.default_rng(seed=2024)
    years = [2019, 2020, 2021, 2022, 2023]
    n_per_industry = 80

    industry_configs = {
        '制造业': {'revenue_mu': 100, 'margin': 0.08, 'turnover': 0.6, 'leverage': 0.55},
        '信息技术': {'revenue_mu': 30, 'margin': 0.18, 'turnover': 0.5, 'leverage': 0.35},
        '金融业': {'revenue_mu': 400, 'margin': 0.28, 'turnover': 0.04, 'leverage': 0.92},
        '消费品': {'revenue_mu': 50, 'margin': 0.12, 'turnover': 0.9, 'leverage': 0.45},
        '能源': {'revenue_mu': 150, 'margin': 0.05, 'turnover': 0.5, 'leverage': 0.65},
    }

    companies = []
    for ind, cfg in industry_configs.items():
        for i in range(n_per_industry):
            companies.append({
                '股票代码': f'{1001 + len(companies):06d}',
                '行业': ind,
                '基础营收': abs(rng.lognormal(np.log(cfg['revenue_mu']), 0.5)),
            })
    df_companies = pd.DataFrame(companies)

    records = []
    for _, comp in df_companies.iterrows():
        cfg = industry_configs[comp['行业']]
        for i, year in enumerate(years):
            rev = comp['基础营收'] * (1 + rng.normal(0.05, 0.1))**i * (1 + rng.normal(0, 0.08))
            margin = cfg['margin'] + rng.normal(0, 0.02)
            cost = rev * (1 - margin)
            net_profit = rev * margin * (1 - 0.25)
            total_assets = rev / cfg['turnover'] * (1 + rng.normal(0, 0.1))
            equity = total_assets * (1 - cfg['leverage'])
            rd_expense = rev * rng.uniform(0.01, 0.15) if comp['行业'] != '金融业' else rev * 0.005
            records.append({
                '股票代码': comp['股票代码'], '行业': comp['行业'], '年份': year,
                '营业收入': rev, '营业成本': cost, '净利润': net_profit,
                '毛利润': rev - cost,
                '归属于母公司股东的净利润': net_profit * rng.uniform(0.7, 1.0),
                '资产总计': total_assets, '负债合计': total_assets * cfg['leverage'],
                '所有者权益合计': equity,
                '研发费用': rd_expense,
                '销售费用': rev * rng.uniform(0.05, 0.12),
                '管理费用': rev * rng.uniform(0.03, 0.08),
                '财务费用': rev * rng.uniform(0.01, 0.05),
                '应收账款': rev * rng.uniform(0.1, 0.3),
                '存货': cost * rng.uniform(0.15, 0.4),
                '流动资产合计': total_assets * rng.uniform(0.4, 0.7),
                '流动负债合计': total_assets * cfg['leverage'] * rng.uniform(0.3, 0.6),
                '货币资金': total_assets * rng.uniform(0.1, 0.3),
                '固定资产': total_assets * rng.uniform(0.2, 0.5),
                '短期借款': total_assets * cfg['leverage'] * rng.uniform(0.3, 0.6),
                '长期借款': total_assets * cfg['leverage'] * rng.uniform(0.2, 0.5),
                '应付账款': cost * rng.uniform(0.2, 0.4),
                '经营活动净现金流': net_profit * rng.uniform(0.8, 1.3),
                '投资活动净现金流': -rev * rng.uniform(0.05, 0.15),
                '筹资活动净现金流': total_assets * cfg['leverage'] * rng.uniform(-0.1, 0.1),
                '现金及等价物净增加额': net_profit * rng.uniform(-0.2, 0.2),
                '公司名称': f"{comp['行业']}{comp['股票代码']}号",
            })
    return pd.DataFrame(records)


# ======================== 财务指标计算 ========================

def compute_ratios(df):
    """计算ROE、资产负债率、总资产周转率"""
    df = df.copy()
    df['ROE'] = np.where(df['所有者权益合计'] != 0,
                         df['归属于母公司股东的净利润'] / df['所有者权益合计'] * 100, np.nan)
    df['ROA'] = np.where(df['资产总计'] != 0,
                         df['净利润'] / df['资产总计'] * 100, np.nan)
    df['资产负债率'] = np.where(df['资产总计'] != 0,
                               df['负债合计'] / df['资产总计'] * 100, np.nan)
    df['总资产周转率'] = np.where(df['资产总计'] != 0,
                                 df['营业收入'] / df['资产总计'], np.nan)
    df['研发强度'] = np.where(df['营业收入'] != 0,
                             df['研发费用'] / df['营业收入'] * 100, 0)
    df['ln总资产'] = np.log(df['资产总计'].clip(lower=1))
    return df


# ======================== AB测试核心逻辑 ========================

def define_treatment_groups(df):
    """步骤1：按研发强度分组（行业年度中位数为界）"""
    # 计算每个行业-年度的研发强度中位数
    industry_year_median = df.groupby(['行业', '年份'])['研发强度'].transform('median')
    df['分组'] = np.where(df['研发强度'] > industry_year_median, '高研发组', '低研发组')
    print("\n【分组结果】")
    print(df['分组'].value_counts())
    print("\n各行业高研发组占比:")
    print(df.groupby('行业')['分组'].value_counts(normalize=True).unstack().round(3))
    return df


def run_ttest(group1, group2, group1_name='处理组', group2_name='对照组'):
    """步骤3：Welch's双样本T检验"""
    print(f"\n【T检验：{group1_name} vs {group2_name}】")
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(), group2.std()
    print(f"  {group1_name}: 均值={mean1:.4f}, 标准差={std1:.4f}, 样本量={n1}")
    print(f"  {group2_name}: 均值={mean2:.4f}, 标准差={std2:.4f}, 样本量={n2}")
    print(f"  均值差: {mean1 - mean2:.4f}")

    # Welch's T检验（不假设方差齐性）
    t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    print(f"  t统计量: {t_stat:.4f}")
    print(f"  p值: {p_value:.6f}")

    # 95%置信区间
    se = np.sqrt(std1**2/n1 + std2**2/n2)
    ci_lower = (mean1 - mean2) - 1.96 * se
    ci_upper = (mean1 - mean2) + 1.96 * se
    print(f"  均值差95%置信区间: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # 显著性判断
    if p_value < 0.01:
        sig = '*** (p<0.01)'
    elif p_value < 0.05:
        sig = '** (p<0.05)'
    elif p_value < 0.1:
        sig = '* (p<0.1)'
    else:
        sig = '不显著 (p>=0.1)'
    print(f"  显著性: {sig}")

    return {'t_stat': t_stat, 'p_value': p_value, 'mean_diff': mean1-mean2, 'ci': (ci_lower, ci_upper),
            'n1': n1, 'n2': n2, 'mean1': mean1, 'mean2': mean2, 'std1': std1, 'std2': std2}


def cohens_d(group1, group2):
    """步骤4：计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    d = (group1.mean() - group2.mean()) / pooled_std if pooled_std != 0 else 0

    # 效应量解释
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = '极小效应（可忽略）'
    elif abs_d < 0.5:
        interpretation = '小效应'
    elif abs_d < 0.8:
        interpretation = '中等效应'
    else:
        interpretation = '大效应'

    print(f"\n【Cohen's d 效应量】")
    print(f"  d = {d:.4f}")
    print(f"  效应大小: {interpretation}")
    print(f"  解释: 高研发组ROE与低研发组ROE的差异为{abs_d:.2f}个标准差")
    return d, interpretation


def propensity_score_matching(df, caliper=0.05):
    """步骤2：倾向得分匹配（PSM）"""
    print("\n【倾向得分匹配（PSM）】")

    # 准备协变量（用于估计倾向得分）
    covariates = ['ln总资产', '资产负债率', '总资产周转率']
    # 行业虚拟变量
    industry_dummies = pd.get_dummies(df['行业'], prefix='行业', drop_first=True)
    X = pd.concat([df[covariates], industry_dummies], axis=1)
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 处理组标签
    y = (df['分组'] == '高研发组').astype(int)

    # Logistic回归估计倾向得分
    logit = LogisticRegression(max_iter=1000, random_state=42)
    logit.fit(X_scaled, y)
    df['倾向得分'] = logit.predict_proba(X_scaled)[:, 1]

    print(f"  倾向得分统计:")
    print(f"    高研发组: 均值={df.loc[y==1, '倾向得分'].mean():.4f}, 标准差={df.loc[y==1, '倾向得分'].std():.4f}")
    print(f"    低研发组: 均值={df.loc[y==0, '倾向得分'].mean():.4f}, 标准差={df.loc[y==0, '倾向得分'].std():.4f}")

    # 1:1近邻匹配（无放回）
    treat_idx = df[df['分组'] == '高研发组'].index.tolist()
    control_idx = df[df['分组'] == '低研发组'].index.tolist()

    treat_scores = df.loc[treat_idx, '倾向得分'].values.reshape(-1, 1)
    control_scores = df.loc[control_idx, '倾向得分'].values.reshape(-1, 1)

    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control_scores)
    distances, indices = nn.kneighbors(treat_scores)

    # 应用卡尺（caliper）过滤
    caliper_threshold = caliper * df['倾向得分'].std()
    matched_treat = []
    matched_control = []
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= caliper_threshold:
            matched_treat.append(treat_idx[i])
            matched_control.append(control_idx[idx])

    print(f"  匹配结果:")
    print(f"    处理组候选: {len(treat_idx)}")
    print(f"    匹配成功: {len(matched_treat)} 对")
    print(f"    卡尺阈值: {caliper_threshold:.4f}")

    # 检查匹配质量（协变量平衡）
    if len(matched_treat) > 0:
        matched_df = df.loc[matched_treat + matched_control]
        print(f"\n  协变量平衡检验（匹配后）:")
        for cov in covariates:
            treat_mean = matched_df.loc[matched_df['分组']=='高研发组', cov].mean()
            control_mean = matched_df.loc[matched_df['分组']=='低研发组', cov].mean()
            print(f"    {cov}: 高研发组={treat_mean:.4f}, 低研发组={control_mean:.4f}")

    return matched_treat, matched_control, df


def balance_test(df):
    """步骤2补充：样本平衡性检验（匹配前）"""
    print("\n【样本平衡性检验】")
    treat = df[df['分组'] == '高研发组']['ln总资产']
    control = df[df['分组'] == '低研发组']['ln总资产']
    t_stat, p_val = stats.ttest_ind(treat, control, equal_var=False)
    print(f"  总资产对数的T检验:")
    print(f"    高研发组均值: {treat.mean():.4f}, 低研发组均值: {control.mean():.4f}")
    print(f"    t={t_stat:.4f}, p={p_val:.6f}")
    if p_val < 0.05:
        print(f"  结论: 两组规模存在显著差异(p={p_val:.4f})，需要PSM匹配")
    else:
        print(f"  结论: 两组规模无显著差异(p={p_val:.4f})，可直接比较")
    return p_val


# ======================== 稳健性检验 ========================

def robustness_checks(df, ttest_result_base, d_base):
    """步骤5：稳健性检验（多种方式验证结论一致性）"""
    print("\n" + "=" * 70)
    print("【稳健性检验】")
    print("=" * 70)
    results = []

    # --- 检验1：不同分组阈值（行业前30% vs 后30%）---
    print("\n【检验1：分组阈值改为前30% vs 后30%】")
    top30_median = df.groupby(['行业', '年份'])['研发强度'].transform(lambda x: x.quantile(0.7))
    bottom30_median = df.groupby(['行业', '年份'])['研发强度'].transform(lambda x: x.quantile(0.3))
    mask_top = df['研发强度'] > top30_median
    mask_bottom = df['研发强度'] < bottom30_median
    roe_top30 = df.loc[mask_top, 'ROE'].dropna()
    roe_bottom30 = df.loc[mask_bottom, 'ROE'].dropna()
    res1 = run_ttest(roe_top30, roe_bottom30, '前30%高研发组', '后30%低研发组')
    d1, interp1 = cohens_d(roe_top30, roe_bottom30)
    results.append({'检验项目': '阈值30%分组', **res1, 'Cohen_d': d1, '效应解释': interp1})

    # --- 检验2：替换被解释变量（ROA代替ROE）---
    print("\n【检验2：替换被解释变量（用ROA）】")
    treat_roa = df.loc[df['分组']=='高研发组', 'ROA'].dropna()
    control_roa = df.loc[df['分组']=='低研发组', 'ROA'].dropna()
    res2 = run_ttest(treat_roa, control_roa, '高研发组-ROA', '低研发组-ROA')
    d2, interp2 = cohens_d(treat_roa, control_roa)
    results.append({'检验项目': '替换ROA', **res2, 'Cohen_d': d2, '效应解释': interp2})

    # --- 检验3：分行业子样本检验 ---
    print("\n【检验3：分行业子样本检验】")
    for industry in df['行业'].unique():
        df_ind = df[df['行业'] == industry]
        treat_ind = df_ind.loc[df_ind['分组']=='高研发组', 'ROE'].dropna()
        control_ind = df_ind.loc[df_ind['分组']=='低研发组', 'ROE'].dropna()
        if len(treat_ind) > 5 and len(control_ind) > 5:
            t_stat, p_val = stats.ttest_ind(treat_ind, control_ind, equal_var=False)
            d_ind, _ = cohens_d(treat_ind, control_ind)
            print(f"    {industry}: t={t_stat:.3f}, p={p_val:.4f}, d={d_ind:.3f}, "
                  f"高研发均值={treat_ind.mean():.3f}, 低研发均值={control_ind.mean():.3f}")

    return results


# ======================== 保存结果 ========================

def save_results(ttest_result, d, interp, robustness_results):
    """步骤6：输出统计检验结果表CSV"""
    rows = [{
        '检验项目': '主检验（匹配后）',
        '样本量(处理组)': ttest_result['n1'],
        '样本量(对照组)': ttest_result['n2'],
        '均值差': round(ttest_result['mean_diff'], 4),
        't统计量': round(ttest_result['t_stat'], 4),
        'p值': round(ttest_result['p_value'], 6),
        'Cohen_d': round(d, 4),
        '效应解释': interp,
    }]
    for r in robustness_results:
        rows.append({
            '检验项目': r.get('检验项目', ''),
            '样本量(处理组)': r.get('n1', ''),
            '样本量(对照组)': r.get('n2', ''),
            '均值差': round(r.get('mean_diff', 0), 4),
            't统计量': round(r.get('t_stat', 0), 4),
            'p值': round(r.get('p_value', 1), 6),
            'Cohen_d': round(r.get('Cohen_d', 0), 4),
            '效应解释': r.get('效应解释', ''),
        })
    df_results = pd.DataFrame(rows)
import os as _os
_output_dir = _os.path.dirname(_os.path.abspath(__file__))
    df_results.to_csv(_os.path.join(_output_dir, '统计检验结果表.csv'), index=False, encoding='utf-8-sig')
    print("\n【结果已保存】统计检验结果表.csv")
    print(df_results.to_string(index=False))
    return df_results


# ======================== 业务解读 ========================

def business_interpretation():
    """步骤7：撰写200字业务解读"""
    print("\n" + "=" * 70)
    print("【业务解读：对制造业CFO的启示】")
    print("=" * 70)
    text = """
作为制造业公司的CFO，本检验的核心结论对研发预算决策有以下启示：

1. 研发投入与ROE的关联性：如果统计检验显示高研发组ROE显著高于低研发组，说明研发投入
   确实是提升盈利能力的有效途径。但这不意味着简单地增加研发预算就能立即见效——
   研发成果转化为利润通常需要2-3年的滞后期。

2. 研发强度的"门槛效应"：稳健性检验（前30% vs 后30%分组）的结果至关重要。如果
   只有超过行业较高阈值的研发投入才显著提升ROE，则意味着"平均化"的研发投入
   可能收效甚微，需要集中资源重点投入。

3. 行业差异：分行业子样本检验可以帮助判断研发对ROE的影响是否因行业而异。
   制造业可能需要更高强度的研发投入才能见到效果，而信息技术行业可能阈值更低。

建议：CFO不应盲目增加研发预算，而应（1）将研发强度提升至行业75分位以上，（2）
确保研发方向与公司战略和市场需求对齐，（3）建立研发绩效跟踪机制，（4）控制
研发支出的风险敞口，避免研发投入挤占其他重要支出。
"""
    print(text)


# ======================== 主程序 ========================

if __name__ == '__main__':
    print("=" * 70)
    print("AB测试：研发投入对ROE的影响检验")
    print("=" * 70)

    # 1. 生成数据
    df = generate_all_data()
    print(f"原始数据维度: {df.shape}")

    # 2. 计算财务指标
    df = compute_ratios(df)

    # 3. 分组
    df = define_treatment_groups(df)

    # 4. 平衡性检验
    balance_p = balance_test(df)

    # 5. PSM匹配
    matched_treat, matched_control, df = propensity_score_matching(df, caliper=0.05)

    # 6. 匹配后T检验
    roe_treat_matched = df.loc[matched_treat, 'ROE'].dropna()
    roe_control_matched = df.loc[matched_control, 'ROE'].dropna()
    print(f"\n匹配后样本: 高研发组 {len(roe_treat_matched)}, 低研发组 {len(roe_control_matched)}")

    # 主检验
    ttest_result = run_ttest(roe_treat_matched, roe_control_matched, '匹配后高研发组', '匹配后低研发组')
    d, interp = cohens_d(roe_treat_matched, roe_control_matched)

    # 显著性结论
    print(f"\n【结论】")
    if ttest_result['p_value'] < 0.05:
        if ttest_result['mean_diff'] > 0:
            print(f"  在α=0.05水平下，高研发组的ROE显著高于低研发组")
        else:
            print(f"  在α=0.05水平下，高研发组的ROE显著低于低研发组")
    else:
        print(f"  在α=0.05水平下，高研发组与低研发组的ROE无显著差异")

    # 7. 稳健性检验
    robustness = robustness_checks(df, ttest_result, d)

    # 8. 保存结果
    save_results(ttest_result, d, interp, robustness)

    # 9. 业务解读
    business_interpretation()
