# -*- coding: utf-8 -*-
"""
02_指标计算与可视化代码.py
==========================
A股上市公司财务指标计算 + Plotly Dash 交互式仪表板
功能：计算ROE/ROA等10个核心财务比率，构建5个交互式图表
作者：数据分析作品集 | 2024
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output

# ======================== 第一部分：数据读取（优先读模块一CSV，否则自生成） ========================

def load_or_generate_data():
    """读取模块一的清洗后数据（cleaned_financial_data.csv），
    若文件不存在则自动生成"""
    import os
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cleaned_financial_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        # 统一列名（模块一用"会计年度"，本模块用"年份"）
        if '会计年度' in df.columns and '年份' not in df.columns:
            df = df.rename(columns={'会计年度': '年份'})
        # 统一列名（模块一用"归母净利润"，本模块用"归属于母公司股东的净利润"）
        if '归母净利润' in df.columns and '归属于母公司股东的净利润' not in df.columns:
            df = df.rename(columns={'归母净利润': '归属于母公司股东的净利润'})
        return df
    return None


def generate_all_data():
    """生成模拟数据（复用模块一的种子和逻辑，确保一致性）"""
    rng = np.random.default_rng(seed=2024)
    years = [2019, 2020, 2021, 2022, 2023]
    n_companies_per_industry = 80
    n_industries = 5

    # 定义行业特征（收入范围、利润率、周转率、杠杆率）
    industry_configs = {
        '制造业': {'revenue_mu': 100, 'revenue_sig': 50, 'margin': 0.08, 'turnover': 0.6, 'leverage': 0.55},
        '信息技术': {'revenue_mu': 30, 'revenue_sig': 15, 'margin': 0.18, 'turnover': 0.5, 'leverage': 0.35},
        '金融业': {'revenue_mu': 400, 'revenue_sig': 200, 'margin': 0.28, 'turnover': 0.04, 'leverage': 0.92},
        '消费品': {'revenue_mu': 50, 'revenue_sig': 25, 'margin': 0.12, 'turnover': 0.9, 'leverage': 0.45},
        '能源': {'revenue_mu': 150, 'revenue_sig': 100, 'margin': 0.05, 'turnover': 0.5, 'leverage': 0.65},
    }

    # 为每家公司生成基础特征
    companies = []
    company_id = 1001
    for ind_name, config in industry_configs.items():
        for _ in range(n_companies_per_industry):
            base_revenue = abs(rng.lognormal(mean=np.log(config['revenue_mu']), sigma=config['revenue_sig']/config['revenue_mu']))
            companies.append({
                '股票代码': f'{company_id:06d}',
                '公司名称': f'{ind_name}{company_id}号',
                '行业': ind_name,
                '基础营收': base_revenue,
                '边际波动': rng.normal(0, 0.08),  # 年度间波动幅度
            })
            company_id += 1
    df_companies = pd.DataFrame(companies)

    # 生成5年数据
    records_income = []
    records_balance = []
    records_cash = []

    for _, comp in df_companies.iterrows():
        config = industry_configs[comp['行业']]
        rev_base = comp['基础营收']
        rev_growth = rng.normal(0.05, 0.1)  # 基础增长率

        for i, year in enumerate(years):
            # --- 营业收入（含年度波动和趋势增长）---
            rev = rev_base * (1 + rev_growth)**i * (1 + comp['边际波动'] * rng.uniform(0.8, 1.2))
            rev = max(rev, 10)  # 确保不为负或过小

            # --- 成本与利润
            margin = config['margin'] + rng.normal(0, 0.02)
            cost = rev * (1 - margin)
            sales_expense = rev * rng.uniform(0.05, 0.12)
            admin_expense = rev * rng.uniform(0.03, 0.08)
            rd_expense = rev * (0.02 if comp['行业'] == '制造业' else rng.uniform(0, 0.15))
            fin_expense = rev * rng.uniform(0.01, 0.05)
            operating_profit = rev - cost - sales_expense - admin_expense - rd_expense - fin_expense
            total_profit = operating_profit * (1 + rng.normal(0, 0.02))
            tax = max(total_profit * 0.25, 0)
            net_profit = total_profit - tax

            # --- 资产负债
            total_assets = rev / config['turnover'] * (1 + rng.normal(0, 0.1))
            leverage = config['leverage'] + rng.normal(0, 0.05)
            leverage = np.clip(leverage, 0.2, 0.95)
            total_debt = total_assets * leverage
            equity = total_assets - total_debt
            current_assets = total_assets * rng.uniform(0.4, 0.7)
            current_liabilities = total_debt * rng.uniform(0.3, 0.6)
            fixed_assets = total_assets * rng.uniform(0.2, 0.5)
            receivables = rev * rng.uniform(0.1, 0.3)
            inventory = cost * rng.uniform(0.15, 0.4)

            # --- 现金流
            operating_cf = net_profit * rng.uniform(0.8, 1.3)
            investing_cf = -rev * rng.uniform(0.05, 0.15)
            financing_cf = total_debt * rng.uniform(-0.1, 0.1)
            cash_change = operating_cf + investing_cf + financing_cf

            # 保存记录
            records_income.append({
                '股票代码': comp['股票代码'], '公司名称': comp['公司名称'], '行业': comp['行业'], '年份': year,
                '营业收入': rev, '营业成本': cost, '毛利润': rev - cost,
                '销售费用': sales_expense, '管理费用': admin_expense, '研发费用': rd_expense,
                '财务费用': fin_expense, '营业利润': operating_profit, '利润总额': total_profit,
                '所得税费用': tax, '净利润': net_profit,
                '归属于母公司股东的净利润': net_profit * rng.uniform(0.7, 1.0),
            })
            records_balance.append({
                '股票代码': comp['股票代码'], '公司名称': comp['公司名称'], '行业': comp['行业'], '年份': year,
                '货币资金': total_assets * rng.uniform(0.1, 0.3),
                '应收账款': receivables, '存货': inventory,
                '流动资产合计': current_assets, '固定资产': fixed_assets,
                '无形资产': total_assets * rng.uniform(0.05, 0.15),
                '非流动资产合计': total_assets - current_assets,
                '资产总计': total_assets, '短期借款': total_debt * rng.uniform(0.3, 0.6),
                '应付账款': cost * rng.uniform(0.2, 0.4), '流动负债合计': current_liabilities,
                '长期借款': total_debt * rng.uniform(0.2, 0.5),
                '非流动负债合计': total_debt - current_liabilities,
                '负债合计': total_debt, '所有者权益合计': equity,
                '归属于母公司股东的权益': equity * rng.uniform(0.6, 1.0),
            })
            records_cash.append({
                '股票代码': comp['股票代码'], '公司名称': comp['公司名称'], '行业': comp['行业'], '年份': year,
                '经营活动现金流入': rev * rng.uniform(1.0, 1.2),
                '经营活动现金流出': cost + sales_expense + admin_expense,
                '经营活动净现金流': operating_cf,
                '投资活动净现金流': investing_cf,
                '筹资活动净现金流': financing_cf,
                '现金及等价物净增加额': cash_change,
            })

    return pd.DataFrame(records_income), pd.DataFrame(records_balance), pd.DataFrame(records_cash)


# ======================== 第二部分：数据清洗（与模块一一致） ========================

def inject_quality_issues(df_income, df_balance, df_cash, rng):
    """向数据注入真实的数据质量问题"""
    for df in [df_income, df_balance, df_cash]:
        n = len(df)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 缺失值（3%）
        for col in numeric_cols:
            mask = rng.random(n) < 0.03
            df.loc[mask, col] = np.nan

        # 异常值（1%）
        outlier_idx = rng.choice(n, size=int(n * 0.01), replace=False)
        if '营业收入' in df.columns:
            df.loc[outlier_idx, '营业收入'] *= 10

        # 标签不一致
        label_issues = {'制造业': '制造', '信息技术': '信息产业'}
        for original, variant in label_issues.items():
            mask = rng.random(n) < 0.03
            df.loc[mask & (df['行业'] == original), '行业'] = variant

    # 研发费用金融业缺失
    mask_fin = df_income['行业'] == '金融业'
    df_income.loc[mask_fin, '研发费用'] = np.nan

    # 重复行
    dup_income = df_income.sample(frac=0.02, random_state=rng)
    df_income = pd.concat([df_income, dup_income], ignore_index=True)
    dup_balance = df_balance.sample(frac=0.02, random_state=rng)
    df_balance = pd.concat([df_balance, dup_balance], ignore_index=True)
    dup_cash = df_cash.sample(frac=0.02, random_state=rng)
    df_cash = pd.concat([df_cash, dup_cash], ignore_index=True)

    return df_income.sample(frac=1, random_state=rng).reset_index(drop=True), \
           df_balance.sample(frac=1, random_state=rng).reset_index(drop=True), \
           df_cash.sample(frac=1, random_state=rng).reset_index(drop=True)


def clean_data(df_income, df_balance, df_cash):
    """执行数据清洗流水线（6步，每步打印对比）"""
    rng = np.random.default_rng(seed=2024)

    # === 步骤1：去重 ===
    print("\n【步骤1：去重】")
    print(f"  利润表：去重前 {len(df_income)} 行, 重复 {df_income.duplicated().sum()} 行")
    df_income = df_income.drop_duplicates()
    print(f"  利润表：去重后 {len(df_income)} 行")
    print(f"  资产负债表：去重前 {len(df_balance)} 行, 重复 {df_balance.duplicated().sum()} 行")
    df_balance = df_balance.drop_duplicates()
    print(f"  资产负债表：去重后 {len(df_balance)} 行")
    print(f"  现金流量表：去重前 {len(df_cash)} 行, 重复 {df_cash.duplicated().sum()} 行")
    df_cash = df_cash.drop_duplicates()
    print(f"  现金流量表：去重后 {len(df_cash)} 行")

    # === 步骤2：行业标签标准化 ===
    print("\n【步骤2：行业标签标准化】")
    label_map = {'制造': '制造业', 'IT': '信息技术', '信息产业': '信息技术', '消费': '消费品', '能': '能源'}
    for df in [df_income, df_balance, df_cash]:
        df['行业'] = df['行业'].replace(label_map)
    print("  修复后行业分布：", df_income['行业'].value_counts().to_dict())

    # === 步骤3：类型转换 ===
    print("\n【步骤3：类型转换（字符串转数值）】")
    coerce_count = 0
    for df in [df_income, df_balance, df_cash]:
        numeric_cols = df.select_dtypes(include='object').columns.difference(['股票代码', '公司名称', '行业', '年份']).tolist()
        for col in numeric_cols:
            before_nan = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            after_nan = df[col].isna().sum()
            coerce_count += after_nan - before_nan
    print(f"  因类型转换新增的缺失值: {coerce_count}")

    # === 步骤4：缺失值填补（4层策略）===
    print("\n【步骤4：缺失值填补】")
    for df, name in [(df_income, '利润表'), (df_balance, '资产负债表'), (df_cash, '现金流量表')]:
        before_total = df.isna().sum().sum()
        # 策略1：研发费用金融业填0
        if '研发费用' in df.columns:
            mask_fin = df['行业'] == '金融业'
            df.loc[mask_fin & df['研发费用'].isna(), '研发费用'] = 0
        # 策略2：数值列用行业年度中位数
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df.groupby(['行业', '年份'])[col].transform(lambda x: x.fillna(x.median()))
        # 策略3：剩余用行业整体中位数
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df.groupby('行业')[col].transform(lambda x: x.fillna(x.median()))
        # 策略4：最后用0填补
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
        after_total = df.isna().sum().sum()
        print(f"  {name}：填补前缺失 {before_total} → 填补后缺失 {after_total}")

    # === 步骤5：缩尾处理（1%/99%）===
    print("\n【步骤5：缩尾处理（1%/99%分位）】")
    winsorize_cols = ['营业收入', '净利润', '总资产', '总负债', '所有者权益合计', '经营活动净现金流']
    clip_counts = {}
    for col in winsorize_cols:
        if col in df_income.columns:
            vals = df_income[col]
        elif col in df_balance.columns:
            vals = df_balance[col]
        else:
            continue
        q01, q99 = vals.quantile(0.01), vals.quantile(0.99)
        clipped = ((vals < q01) | (vals > q99)).sum()
        clip_counts[col] = clipped
        if col in df_income.columns:
            df_income[col] = df_income[col].clip(lower=q01, upper=q99)
        elif col in df_balance.columns:
            df_balance[col] = df_balance[col].clip(lower=q01, upper=q99)
    print("  缩尾处理统计：", clip_counts)

    # === 步骤6：合并为宽表 ===
    print("\n【步骤6：三表合并】")
    key_cols = ['股票代码', '公司名称', '行业', '年份']
    merged = df_income.merge(df_balance, on=key_cols, how='outer', suffixes=('_利润', '_资产'))
    merged = merged.merge(df_cash, on=key_cols, how='outer', suffixes=('', '_现金'))
    print(f"  合并后宽表维度: {merged.shape}")
    print(f"  前5行预览:\n{merged.head()}")
    return merged


# ======================== 第三部分：财务指标计算 ========================

def compute_financial_ratios(df):
    """计算10个核心财务指标，处理分母为零的情况"""
    print("\n【计算财务指标】")
    df = df.copy()

    # --- 盈利能力指标 ---
    # ROE = 净利润 / 所有者权益
    df['ROE'] = np.where(df['所有者权益合计'] != 0,
                         df['归属于母公司股东的净利润'] / df['所有者权益合计'] * 100, np.nan)
    # ROA = 净利润 / 总资产
    df['ROA'] = np.where(df['资产总计'] != 0,
                         df['净利润'] / df['资产总计'] * 100, np.nan)
    # 毛利率 = 毛利润 / 营业收入
    df['毛利率'] = np.where(df['营业收入'] != 0,
                            df['毛利润'] / df['营业收入'] * 100, np.nan)
    # 净利率 = 净利润 / 营业收入
    df['净利率'] = np.where(df['营业收入'] != 0,
                           df['净利润'] / df['营业收入'] * 100, np.nan)

    # --- 偿债能力指标 ---
    # 资产负债率 = 总负债 / 总资产
    df['资产负债率'] = np.where(df['资产总计'] != 0,
                               df['负债合计'] / df['资产总计'] * 100, np.nan)
    # 流动比率 = 流动资产 / 流动负债
    df['流动比率'] = np.where(df['流动负债合计'] != 0,
                             df['流动资产合计'] / df['流动负债合计'], np.nan)

    # --- 营运能力指标 ---
    # 总资产周转率 = 营业收入 / 平均总资产
    df = df.sort_values(['股票代码', '年份'])
    df['平均总资产'] = df.groupby('股票代码')['资产总计'].transform(lambda x: (x + x.shift(1)) / 2)
    df['总资产周转率'] = np.where(df['平均总资产'] != 0,
                                 df['营业收入'] / df['平均总资产'], np.nan)
    # 应收账款周转率
    df['平均应收账款'] = df.groupby('股票代码')['应收账款'].transform(lambda x: (x + x.shift(1)) / 2)
    df['应收账款周转率'] = np.where(df['平均应收账款'] != 0,
                                  df['营业收入'] / df['平均应收账款'], np.nan)
    # 存货周转率
    df['平均存货'] = df.groupby('股票代码')['存货'].transform(lambda x: (x + x.shift(1)) / 2)
    df['存货周转率'] = np.where(df['平均存货'] != 0,
                               df['营业成本'] / df['平均存货'], np.nan)

    # --- 成长能力指标 ---
    df['营收增长率'] = df.groupby('股票代码')['营业收入'].pct_change() * 100
    df['净利润增长率'] = df.groupby('股票代码')['净利润'].pct_change() * 100

    # 打印各指标描述性统计（按行业）
    ratio_cols = ['ROE', 'ROA', '毛利率', '净利率', '资产负债率', '总资产周转率', '应收账款周转率', '存货周转率']
    print("\n【各指标按行业描述性统计】")
    for ratio in ratio_cols:
        if ratio in df.columns:
            stats = df.groupby('行业')[ratio].agg(['mean', 'std', 'min', 'max']).round(2)
            print(f"\n  --- {ratio} ---")
            print(stats.to_string(index=True))

    return df


# ======================== 第四部分：Dash 交互式仪表板 ========================

def create_dash_app(df):
    """创建Plotly Dash交互式仪表板，包含5个图表组件"""
    app = Dash(__name__)

    # 准备下拉菜单选项
    year_options = [{'label': str(y), 'value': y} for y in sorted(df['年份'].unique())]
    metric_options = [
        {'label': 'ROE（净资产收益率）', 'value': 'ROE'},
        {'label': 'ROA（总资产收益率）', 'value': 'ROA'},
        {'label': '毛利率', 'value': '毛利率'},
        {'label': '净利率', 'value': '净利率'},
    ]
    company_options = [{'label': f"{row['股票代码']} {row['公司名称']}", 'value': row['股票代码']}
                       for _, row in df.drop_duplicates('股票代码')[['股票代码', '公司名称']].iterrows()]

    app.layout = html.Div([
        html.H1("A股上市公司基本面分析仪表板", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),

        # --- 第1行：KPI 卡片 ---
        html.Div(id='kpi-cards', style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 40,
                                        'backgroundColor': '#f8f9fa', 'padding': 20, 'borderRadius': 10}),

        # --- 第2行：散点图 + 时序图 ---
        html.Div([
            html.Div([
                html.H3("行业对比散点图", style={'textAlign': 'center'}),
                html.Label("选择年份："),
                dcc.Dropdown(id='year-dropdown', options=year_options, value=year_options[-1]['value'],
                            style={'width': '200px', 'marginBottom': 10}),
                dcc.Graph(id='scatter-graph'),
            ], style={'width': '48%'}),
            html.Div([
                html.H3("行业ROE时序走势", style={'textAlign': 'center'}),
                html.Label("选择指标："),
                dcc.Dropdown(id='metric-dropdown', options=metric_options, value='ROE',
                            style={'width': '250px', 'marginBottom': 10}),
                dcc.Graph(id='timeseries-graph'),
            ], style={'width': '48%'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': 40}),

        # --- 第3行：公司筛选 + 热力图 ---
        html.Div([
            html.Div([
                html.H3("公司财务画像", style={'textAlign': 'center'}),
                html.Label("选择公司："),
                dcc.Dropdown(id='company-dropdown', options=company_options, value=company_options[0]['value'],
                            style={'width': '300px', 'marginBottom': 10}),
                dcc.Graph(id='radar-graph'),
            ], style={'width': '48%'}),
            html.Div([
                html.H3("行业指标热力图", style={'textAlign': 'center'}),
                html.Label("选择指标："),
                dcc.Dropdown(id='heatmap-metric', options=metric_options, value='ROE',
                            style={'width': '250px', 'marginBottom': 10}),
                dcc.Graph(id='heatmap-graph'),
            ], style={'width': '48%'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
    ], style={'maxWidth': 1400, 'margin': '0 auto', 'padding': 30})

    # --- 回调函数 ---

    @app.callback(Output('kpi-cards', 'children'), Input('year-dropdown', 'value'))
    def update_kpi(year):
        """更新KPI卡片：显示全市场平均指标"""
        df_year = df[df['年份'] == year]
        avg_roe = df_year['ROE'].mean()
        avg_roa = df_year['ROA'].mean()
        avg_debt = df_year['资产负债率'].mean()
        avg_turnover = df_year['总资产周转率'].mean()
        card_style = {'backgroundColor': 'white', 'padding': 25, 'borderRadius': 8, 'textAlign': 'center',
                      'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '22%'}
        return [
            html.Div([html.H4("平均ROE"), html.P(f"{avg_roe:.2f}%", style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#e74c3c'})], style=card_style),
            html.Div([html.H4("平均ROA"), html.P(f"{avg_roa:.2f}%", style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#3498db'})], style=card_style),
            html.Div([html.H4("平均资产负债率"), html.P(f"{avg_debt:.2f}%", style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#f39c12'})], style=card_style),
            html.Div([html.H4("平均总资产周转率"), html.P(f"{avg_turnover:.2f}", style={'fontSize': 24, 'fontWeight': 'bold', 'color': '#27ae60'})], style=card_style),
        ]

    @app.callback(Output('scatter-graph', 'figure'), Input('year-dropdown', 'value'))
    def update_scatter(year):
        """行业对比散点图：ROE vs 资产负债率，颜色=行业，大小=总资产"""
        df_year = df[df['年份'] == year].dropna(subset=['ROE', '资产负债率', '资产总计'])
        fig = px.scatter(df_year, x='资产负债率', y='ROE', color='行业',
                        size='资产总计', hover_data=['公司名称', '股票代码'],
                        title=f"{year}年 全市场公司ROE vs 资产负债率",
                        labels={'资产负债率': '资产负债率(%)', 'ROE': 'ROE(%)'},
                        size_max=50, opacity=0.7)
        fig.update_layout(height=450)
        return fig

    @app.callback(Output('timeseries-graph', 'figure'), Input('metric-dropdown', 'value'))
    def update_timeseries(metric):
        """时序折线图：各行业指标中位数趋势"""
        metric_names = {'ROE': 'ROE(%)', 'ROA': 'ROA(%)', '毛利率': '毛利率(%)', '净利率': '净利率(%)'}
        industry_year = df.groupby(['年份', '行业'])[metric].median().reset_index()
        fig = px.line(industry_year, x='年份', y=metric, color='行业',
                     markers=True, title=f"近5年各行业{metric_names.get(metric, metric)}中位数走势",
                     labels={'年份': '年份', metric: metric_names.get(metric, metric)})
        fig.update_layout(height=450)
        return fig

    @app.callback(Output('radar-graph', 'figure'), Input('company-dropdown', 'value'))
    def update_radar(company_code):
        """公司雷达图：多维财务指标对比"""
        df_comp = df[df['股票代码'] == company_code]
        if df_comp.empty:
            return go.Figure()
        latest = df_comp[df_comp['年份'] == df_comp['年份'].max()].iloc[0]
        categories = ['ROE', 'ROA', '毛利率', '净利率', '总资产周转率', '应收账款周转率', '存货周转率', '流动比率']
        values = [min(max(latest.get(cat, 0), 0), 100) for cat in categories]
        values.append(values[0])  # 闭合图形
        categories.append(categories[0])
        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', name=latest['公司名称']))
        fig.update_layout(title=f"{latest['公司名称']}（{latest['年份']}年）财务指标雷达图",
                         polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=450)
        return fig

    @app.callback(Output('heatmap-graph', 'figure'), Input('heatmap-metric', 'value'))
    def update_heatmap(metric):
        """热力图：行业×年份 指标对比"""
        pivot = df.groupby(['行业', '年份'])[metric].median().unstack()
        fig = go.Figure(data=go.Heatmap(z=pivot.values, x=[str(y) for y in pivot.columns],
                                        y=pivot.index, colorscale='RdYlGn', text=np.round(pivot.values, 2),
                                        texttemplate='%{text}', textfont={"size": 10}))
        fig.update_layout(title=f"各行业{metric}中位数热力图", height=450)
        return fig

    return app


# ======================== 主程序 ========================

if __name__ == '__main__':
    print("=" * 70)
    print("A股上市公司财务指标计算 + 交互式仪表板")
    print("=" * 70)

    # 1. 优先读取模块一的清洗后数据，不存在则自生成
    df_cleaned = load_or_generate_data()
    if df_cleaned is not None:
        print(f"  已读取模块一清洗后数据: {df_cleaned.shape}")
    else:
        print("  模块一CSV不存在，自动生成数据...")
        rng = np.random.default_rng(seed=2024)
        df_income, df_balance, df_cash = generate_all_data()
        df_income, df_balance, df_cash = inject_quality_issues(df_income, df_balance, df_cash, rng)
        df_cleaned = clean_data(df_income, df_balance, df_cash)

    # 2. 计算财务指标
    df_ratios = compute_financial_ratios(df_cleaned)

    # 4. 输出清洗后数据概览
    print("\n【数据预览】")
    print(f"  最终数据维度: {df_ratios.shape}")
    print(f"  行业分布:\n{df_ratios['行业'].value_counts()}")
    print(f"  年份分布:\n{df_ratios['年份'].value_counts().sort_index()}")

    # 5. 启动仪表板
    print("\n【启动 Dash 仪表板】")
    print("  请在浏览器中打开: http://localhost:8050")
    app = create_dash_app(df_ratios)
    app.run(debug=True, port=8050)
