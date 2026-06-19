# -*- coding: utf-8 -*-
"""生成数据质量报告图表（独立脚本）"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
_output_dir = os.path.dirname(os.path.abspath(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 配置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

print('环境配置完成，开始生成图表...')

# 复现数据
SEED = 2024
YEARS = [2019, 2020, 2021, 2022, 2023]
INDUSTRY_CONFIG = {
    '制造业': {'prefix': '6000', 'n': 80, 'revenue_logmean': 3.0, 'revenue_logstd': 1.0,
              'net_margin': (0.03, 0.12), 'asset_turnover': (0.4, 0.8), 'debt_ratio': (0.40, 0.65), 'rd_intensity': (0.03, 0.08)},
    '信息技术': {'prefix': '3000', 'n': 80, 'revenue_logmean': 2.0, 'revenue_logstd': 1.2,
              'net_margin': (0.08, 0.25), 'asset_turnover': (0.3, 0.7), 'debt_ratio': (0.20, 0.45), 'rd_intensity': (0.10, 0.25)},
    '金融业': {'prefix': '6013', 'n': 80, 'revenue_logmean': 4.0, 'revenue_logstd': 1.5,
              'net_margin': (0.15, 0.35), 'asset_turnover': (0.02, 0.06), 'debt_ratio': (0.85, 0.95), 'rd_intensity': (0.001, 0.01)},
    '消费品': {'prefix': '0008', 'n': 80, 'revenue_logmean': 2.5, 'revenue_logstd': 1.0,
              'net_margin': (0.05, 0.18), 'asset_turnover': (0.6, 1.2), 'debt_ratio': (0.30, 0.55), 'rd_intensity': (0.01, 0.04)},
    '能源': {'prefix': '6005', 'n': 80, 'revenue_logmean': 3.5, 'revenue_logstd': 1.2,
              'net_margin': (0.02, 0.10), 'asset_turnover': (0.3, 0.7), 'debt_ratio': (0.45, 0.75), 'rd_intensity': (0.01, 0.03)},
}

rng = np.random.default_rng(seed=SEED)
records = []
for ind, cfg in INDUSTRY_CONFIG.items():
    for i in range(cfg['n']):
        code = f"{cfg['prefix']}{i:04d}"
        name = f"{ind}第{i+1:03d}号公司"
        base_rev = rng.lognormal(mean=cfg['revenue_logmean'], sigma=cfg['revenue_logstd'])
        base_margin = rng.uniform(*cfg['net_margin'])
        base_turnover = rng.uniform(*cfg['asset_turnover'])
        base_debt = rng.uniform(*cfg['debt_ratio'])
        base_rd = rng.uniform(*cfg['rd_intensity'])
        for year in YEARS:
            shock = rng.normal(0, 0.05)
            growth = 1 + (year - 2019) * 0.03
            rev = max(base_rev * growth * (1 + shock), 0.1)
            cost = rev * (1 - rng.uniform(0.15, 0.45))
            ta = rev / max(base_turnover, 0.01)
            records.append({
                '股票代码': code, '公司名称': name, '行业': ind, '会计年度': year,
                '营业收入': round(rev, 2), '营业成本': round(cost, 2),
                '销售费用': round(rev * rng.uniform(0.02, 0.08), 2),
                '管理费用': round(rev * rng.uniform(0.03, 0.10), 2),
                '研发费用': round(max(rev * base_rd * (1 + rng.normal(0, 0.1)), 0), 2),
                '财务费用': round(rev * rng.uniform(0.005, 0.03), 2),
                '净利润': round(rev * base_margin * (1 + rng.normal(0, 0.1)), 2),
                '归母净利润': round(rev * base_margin * rng.uniform(0.8, 1.0), 2),
                '货币资金': round(ta * rng.uniform(0.08, 0.25), 2),
                '应收账款': round(rev * rng.uniform(0.05, 0.20), 2),
                '存货': round(cost * rng.uniform(0.10, 0.30), 2),
                '资产总计': round(ta, 2),
                '负债合计': round(ta * base_debt, 2),
                '所有者权益合计': round(ta * (1 - base_debt), 2),
                '归母股东权益': round(ta * (1 - base_debt) * rng.uniform(0.85, 1.0), 2),
                '经营活动现金流净额': round(rev * rng.uniform(0.05, 0.15), 2),
                '投资活动现金流净额': round(-ta * rng.uniform(0.02, 0.08), 2),
                '筹资活动现金流净额': round(rng.normal(0, rev * 0.05), 2),
                '现金及等价物净增加额': round(rng.normal(0, rev * 0.05), 2),
            })

df_dirty = pd.DataFrame(records)
n = len(df_dirty)
numeric_cols = ['营业收入', '营业成本', '销售费用', '管理费用', '研发费用', '财务费用',
                '净利润', '归母净利润', '货币资金', '应收账款', '存货', '资产总计',
                '负债合计', '所有者权益合计', '经营活动现金流净额', '筹资活动现金流净额']
for col in numeric_cols:
    mask = rng.random(n) < 0.03
    df_dirty.loc[mask, col] = np.nan
df_dirty.loc[df_dirty['行业'] == '金融业', '研发费用'] = np.nan

df_before = df_dirty.copy()
df_clean = df_dirty.copy()
df_clean.loc[(df_clean['行业'] == '金融业') & df_clean['研发费用'].isna(), '研发费用'] = 0
for col in numeric_cols:
    if col in df_clean.columns:
        median_vals = df_clean.groupby(['行业', '会计年度'])[col].transform('median')
        df_clean[col] = df_clean[col].fillna(median_vals)
for col in ['营业收入', '营业成本', '净利润', '资产总计', '负债合计']:
    q01 = df_clean[col].quantile(0.01)
    q99 = df_clean[col].quantile(0.99)
    df_clean[col] = df_clean[col].clip(lower=q01, upper=q99)

print(f'数据准备完成: 脏数据{df_before.shape}, 清洗后{df_clean.shape}')

# ======================== 图1：缺失率对比 ========================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
missing_before = df_before[numeric_cols].isnull().mean() * 100
missing_after = df_clean[numeric_cols].isnull().mean() * 100

colors_b = ['#ff6b6b' if v > 5 else '#ffd93d' if v > 0 else '#6bcb77' for v in missing_before.values]
axes[0].barh(range(len(numeric_cols)), missing_before.values, color=colors_b)
axes[0].set_yticks(range(len(numeric_cols)))
axes[0].set_yticklabels(numeric_cols, fontproperties=font_prop, fontsize=10)
axes[0].set_xlabel('缺失率 (%)', fontproperties=font_prop)
axes[0].set_title('清洗前 各字段缺失率', fontproperties=font_prop, fontsize=13, fontweight='bold')
for i, v in enumerate(missing_before.values):
    if v > 0:
        axes[0].text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=8)

colors_a = ['#ff6b6b' if v > 5 else '#ffd93d' if v > 0 else '#6bcb77' for v in missing_after.values]
axes[1].barh(range(len(numeric_cols)), missing_after.values, color=colors_a)
axes[1].set_yticks(range(len(numeric_cols)))
axes[1].set_yticklabels(numeric_cols, fontproperties=font_prop, fontsize=10)
axes[1].set_xlabel('缺失率 (%)', fontproperties=font_prop)
axes[1].set_title('清洗后 各字段缺失率', fontproperties=font_prop, fontsize=13, fontweight='bold')
for i, v in enumerate(missing_after.values):
    if v > 0:
        axes[1].text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=8)

from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor='#ff6b6b', label='高缺失(>5%)'),
                    Patch(facecolor='#ffd93d', label='低缺失(0-5%)'),
                    Patch(facecolor='#6bcb77', label='无缺失')],
           loc='lower center', ncol=3, prop=font_prop)
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(os.path.join(_output_dir, 'fig1_缺失率对比.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图1 缺失率对比 已保存')

# ======================== 图2：行业分布 ========================
fig, ax = plt.subplots(figsize=(10, 6))
ind_counts = df_clean.groupby('行业')['股票代码'].nunique().sort_values()
colors_ind = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4ecdc4', '#45b7d1']
ax.barh(ind_counts.index, ind_counts.values, color=colors_ind)
for i, (ind, v) in enumerate(zip(ind_counts.index, ind_counts.values)):
    ax.text(v + 0.5, i, f'{v}家', va='center', fontproperties=font_prop, fontsize=11)
ax.set_xlabel('公司数量', fontproperties=font_prop)
ax.set_title('各行业公司数量分布', fontproperties=font_prop, fontsize=14, fontweight='bold')
ax.set_yticklabels(ind_counts.index, fontproperties=font_prop)
plt.tight_layout()
plt.savefig(os.path.join(_output_dir, 'fig2_行业分布.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图2 行业分布 已保存')

# ======================== 图3：行业箱线图 ========================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
order = ['制造业', '信息技术', '金融业', '消费品', '能源']
palette = dict(zip(order, colors_ind))
for idx, (col, title) in enumerate(zip(['营业收入', '净利润', '资产总计'], ['营业收入', '净利润', '总资产'])):
    sns.boxplot(data=df_clean, x='行业', y=col, order=order, palette=palette, ax=axes[idx])
    axes[idx].set_title(f'各行业{title}分布', fontproperties=font_prop, fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('行业', fontproperties=font_prop)
    axes[idx].set_ylabel(f'{title}(亿元)', fontproperties=font_prop)
    axes[idx].set_xticklabels(order, fontproperties=font_prop, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(_output_dir, 'fig3_行业箱线图.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图3 行业箱线图 已保存')

# ======================== 图4：缩尾前后对比 ========================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, col in enumerate(['营业收入', '净利润', '资产总计']):
    d_before = df_before[col].dropna()
    d_after = df_clean[col].dropna()
    axes[idx].hist(d_before, bins=40, alpha=0.5, color='#ff6b6b', label='缩尾前', density=True)
    axes[idx].hist(d_after, bins=40, alpha=0.5, color='#4ecdc4', label='缩尾后', density=True)
    axes[idx].set_title(f'{col} 分布对比', fontproperties=font_prop, fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('金额(亿元)', fontproperties=font_prop)
    axes[idx].legend(prop=font_prop)
plt.tight_layout()
plt.savefig(os.path.join(_output_dir, 'fig4_缩尾对比.png'), dpi=150, bbox_inches='tight')
plt.close()
print('图4 缩尾对比 已保存')

# ======================== 打印统计摘要 ========================
print('\n' + '=' * 60)
print('【数据质量报告摘要】')
print('=' * 60)
print(f'总样本量: {len(df_clean)}')
print(f'公司数量: {df_clean["股票代码"].nunique()}')
print(f'行业数量: {df_clean["行业"].nunique()}')
print(f'时间跨度: {df_clean["会计年度"].min()}-{df_clean["会计年度"].max()}')
print(f'字段数量: {len(df_clean.columns)}')
print(f'\n清洗后仍存在缺失的字段:')
for col in df_clean.columns:
    pct = df_clean[col].isnull().mean() * 100
    if pct > 0:
        print(f'  {col}: {pct:.2f}%')
print('\n所有图表已生成完毕！')
