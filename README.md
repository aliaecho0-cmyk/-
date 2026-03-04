[stock_backtest_streamlit.py](https://github.com/user-attachments/files/25735928/stock_backtest_streamlit.py)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime

# 设置页面配置
st.set_page_config(
    page_title="股票回测工具（增强版）",
    page_icon="📈",
    layout="wide"
)

# 常量定义
FEE_RATE = 0.0001  # 万分之一
UNIT_COUNT = 5  # 总资金分成 5 份

# 页面标题
st.title("股票回测工具（增强版）")
st.markdown("支持分仓（5份）、新增策略（Bollinger/KDJ/WR/DMI）、绘制资产/收益曲线")

# 数据导入部分
st.header("数据导入")
uploaded_file = st.file_uploader("上传 CSV 文件", type="csv")

if uploaded_file is not None:
    # 读取CSV文件
    try:
        df = pd.read_csv(uploaded_file)
        # 标准化列名
        df.columns = df.columns.str.strip()
        
        # 检查必要的列
        required_columns = ['日期', '股票代码', '开盘价', '收盘价', '最高价', '最低价', '成交量']
        if all(col in df.columns for col in required_columns):
            # 数据预览
            st.subheader("数据预览")
            st.write(f"数据条数: {len(df)}")
            st.write(f"股票数量: {df['股票代码'].nunique()}")
            st.write(f"日期范围: {df['日期'].min()} ~ {df['日期'].max()}")
            st.dataframe(df.head())
            
            # 参数设置部分
            st.header("参数设置")
            
            # 初始资金
            initial_capital = st.number_input("初始资金", value=100000, step=1000)
            
            # 日期范围
            min_date = pd.to_datetime(df['日期']).min()
            max_date = pd.to_datetime(df['日期']).max()
            start_date = st.date_input("起始日期", min_value=min_date, max_value=max_date, value=min_date)
            end_date = st.date_input("终止日期", min_value=min_date, max_value=max_date, value=max_date)
            
            # 策略选择
            strategy = st.selectbox(
                "策略选择",
                [
                    "自定义表达式（原始）",
                    "5日均线上穿20日均线（MA5/MA20）",
                    "MACD 金叉/死叉",
                    "RSI 超买超卖 (14)",
                    "MACD + RSI 组合",
                    "布林带超跌反弹（Bollinger + KDJ）",
                    "KDJ 钝化+量能博弈",
                    "威廉指标 WR（14）",
                    "DMI 趋势确认突破"
                ]
            )
            
            # 自定义条件（仅当选择自定义策略时显示）
            buy_condition = "close > open"
            sell_condition = "close < open"
            
            if strategy == "自定义表达式（原始）":
                buy_condition = st.text_input("买入条件", value="close > open")
                sell_condition = st.text_input("卖出条件", value="close < open")
                st.caption("可用变量: open, close, high, low, volume, ma5, ma20, macdLine, signalLine, rsi, k, d, j, bb_upper, bb_middle, bb_lower, wr, plusDI, minusDI, adx")
            
            # 运行回测按钮
            if st.button("运行回测"):
                # 数据预处理
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[(df['日期'] >= pd.to_datetime(start_date)) & (df['日期'] <= pd.to_datetime(end_date))]
                
                # 按股票代码分组
                data_by_code = {}
                for code, group in df.groupby('股票代码'):
                    data_by_code[code] = group.sort_values('日期').reset_index(drop=True)
                
                # 计算指标函数
                def compute_indicators(data):
                    # 计算MA5, MA20
                    data['ma5'] = data['收盘价'].rolling(window=5).mean()
                    data['ma20'] = data['收盘价'].rolling(window=20).mean()
                    
                    # 计算MACD
                    exp1 = data['收盘价'].ewm(span=12, adjust=False).mean()
                    exp2 = data['收盘价'].ewm(span=26, adjust=False).mean()
                    data['macdLine'] = exp1 - exp2
                    data['signalLine'] = data['macdLine'].ewm(span=9, adjust=False).mean()
                    
                    # 计算RSI
                    delta = data['收盘价'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    data['rsi'] = 100 - (100 / (1 + rs))
                    
                    # 计算布林带
                    data['bb_middle'] = data['收盘价'].rolling(window=20).mean()
                    data['bb_std'] = data['收盘价'].rolling(window=20).std()
                    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
                    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
                    data['bb_dev'] = (data['收盘价'] - data['bb_middle']) / data['bb_std']
                    
                    # 计算KDJ
                    low = data['最低价'].rolling(window=9).min()
                    high = data['最高价'].rolling(window=9).max()
                    data['rsv'] = (data['收盘价'] - low) / (high - low) * 100
                    data['k'] = data['rsv'].ewm(alpha=1/3, adjust=False).mean()
                    data['d'] = data['k'].ewm(alpha=1/3, adjust=False).mean()
                    data['j'] = 3 * data['k'] - 2 * data['d']
                    
                    # 计算WR
                    data['wr'] = -100 * (high - data['收盘价']) / (high - low)
                    
                    # 计算DMI
                    data['up_move'] = data['最高价'].diff()
                    data['down_move'] = -data['最低价'].diff()
                    data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
                    data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'], 0)
                    data['tr'] = np.maximum(data['最高价'] - data['最低价'], 
                                          np.maximum(abs(data['最高价'] - data['收盘价'].shift(1)), 
                                                    abs(data['最低价'] - data['收盘价'].shift(1))))
                    
                    # 平滑处理
                    data['atr'] = data['tr'].rolling(window=14).mean()
                    data['plus_di'] = 100 * data['plus_dm'].rolling(window=14).mean() / data['atr']
                    data['minus_di'] = 100 * data['minus_dm'].rolling(window=14).mean() / data['atr']
                    data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
                    data['adx'] = data['dx'].rolling(window=14).mean()
                    
                    return data
                
                # 计算所有股票的指标
                for code in data_by_code:
                    data_by_code[code] = compute_indicators(data_by_code[code])
                
                # 策略决策函数
                def strategy_decision(strategy_name, data, idx, state):
                    day = data.iloc[idx]
                    prev = data.iloc[idx-1] if idx > 0 else None
                    remaining_units = UNIT_COUNT - state['units_held']
                    
                    if strategy_name == "自定义表达式（原始）":
                        # 自定义策略逻辑
                        try:
                            buy_cond = eval(buy_condition.replace('close', str(day['收盘价'])).replace('open', str(day['开盘价'])).replace('high', str(day['最高价'])).replace('low', str(day['最低价'])).replace('volume', str(day['成交量'])))
                            sell_cond = eval(sell_condition.replace('close', str(day['收盘价'])).replace('open', str(day['开盘价'])).replace('high', str(day['最高价'])).replace('low', str(day['最低价'])).replace('volume', str(day['成交量'])))
                            return {'buy': 1 if buy_cond else 0, 'sell': state['units_held'] if sell_cond else 0, 'reason': 'custom'}
                        except:
                            return {'buy': 0, 'sell': 0, 'reason': 'error'}
                    
                    elif strategy_name == "5日均线上穿20日均线（MA5/MA20）":
                        if prev is not None and not pd.isna(day['ma5']) and not pd.isna(day['ma20']) and not pd.isna(prev['ma5']) and not pd.isna(prev['ma20']):
                            if prev['ma5'] <= prev['ma20'] and day['ma5'] > day['ma20']:
                                return {'buy': 1, 'sell': 0, 'reason': 'ma_crossover'}
                            elif prev['ma5'] >= prev['ma20'] and day['ma5'] < day['ma20']:
                                return {'buy': 0, 'sell': state['units_held'], 'reason': 'ma_crossover'}
                    
                    elif strategy_name == "MACD 金叉/死叉":
                        if prev is not None and not pd.isna(day['macdLine']) and not pd.isna(day['signalLine']) and not pd.isna(prev['macdLine']) and not pd.isna(prev['signalLine']):
                            if prev['macdLine'] <= prev['signalLine'] and day['macdLine'] > day['signalLine']:
                                return {'buy': 1, 'sell': 0, 'reason': 'macd_crossover'}
                            elif prev['macdLine'] >= prev['signalLine'] and day['macdLine'] < day['signalLine']:
                                return {'buy': 0, 'sell': state['units_held'], 'reason': 'macd_crossover'}
                    
                    elif strategy_name == "RSI 超买超卖 (14)":
                        if not pd.isna(day['rsi']):
                            if day['rsi'] < 30:
                                return {'buy': 1, 'sell': 0, 'reason': 'rsi'}
                            elif day['rsi'] > 70:
                                return {'buy': 0, 'sell': state['units_held'], 'reason': 'rsi'}
                    
                    elif strategy_name == "MACD + RSI 组合":
                        macd_buy = False
                        rsi_buy = False
                        macd_sell = False
                        rsi_sell = False
                        
                        if prev is not None and not pd.isna(day['macdLine']) and not pd.isna(day['signalLine']) and not pd.isna(prev['macdLine']) and not pd.isna(prev['signalLine']):
                            macd_buy = prev['macdLine'] <= prev['signalLine'] and day['macdLine'] > day['signalLine']
                            macd_sell = prev['macdLine'] >= prev['signalLine'] and day['macdLine'] < day['signalLine']
                        
                        if not pd.isna(day['rsi']):
                            rsi_buy = day['rsi'] < 40
                            rsi_sell = day['rsi'] > 60
                        
                        if macd_buy and rsi_buy:
                            return {'buy': 1, 'sell': 0, 'reason': 'macd_rsi'}
                        elif macd_sell or rsi_sell:
                            return {'buy': 0, 'sell': state['units_held'], 'reason': 'macd_rsi'}
                    
                    elif strategy_name == "布林带超跌反弹（Bollinger + KDJ）":
                        buy_count = 0
                        if not pd.isna(day['bb_lower']) and day['收盘价'] < day['bb_lower'] and not pd.isna(day['j']) and day['j'] < -10:
                            buy_count = 1
                            if prev is not None and not pd.isna(prev['bb_lower']) and prev['收盘价'] < prev['bb_lower'] and not pd.isna(day['bb_dev']) and not pd.isna(prev['bb_dev']) and day['bb_dev'] < prev['bb_dev']:
                                buy_count = 2
                        
                        sell_count = 0
                        if not pd.isna(day['bb_middle']) and day['收盘价'] >= day['bb_middle'] and prev is not None:
                            bottom_count = len([u for u in state['unit_records'] if u['tag'] == 'bottom'])
                            sell_count = bottom_count
                        
                        buy_count = min(buy_count, remaining_units)
                        sell_count = min(sell_count, state['units_held'])
                        return {'buy': buy_count, 'sell': sell_count, 'reason': 'bollinger_kdj'}
                    
                    elif strategy_name == "KDJ 钝化+量能博弈":
                        buy_count = 0
                        lookback = 60
                        start = max(0, idx - lookback + 1)
                        slice_data = data.iloc[start:idx+1]
                        max_vol = slice_data['成交量'].max()
                        vol_threshold = max_vol / 3
                        
                        last3 = data.iloc[max(0, idx-2):idx+1]
                        k_flat = False
                        if len(last3) == 3 and not last3['k'].isna().any():
                            diffs = [last3.iloc[1]['k'] - last3.iloc[0]['k'], last3.iloc[2]['k'] - last3.iloc[1]['k']]
                            k_flat = all(d >= -2 for d in diffs)
                        
                        if not pd.isna(day['k']) and day['k'] < 20 and k_flat:
                            recent_vol_small = max_vol > 0 and day['成交量'] <= vol_threshold
                            vol_break = prev is not None and day['成交量'] > prev['成交量'] * 1.5 and day['收盘价'] > prev['收盘价']
                            if recent_vol_small and vol_break:
                                buy_count = 2 if remaining_units >= 2 else (1 if remaining_units >= 1 else 0)
                        
                        sell_count = 0
                        if not pd.isna(day['k']) and day['k'] >= 80 and prev is not None and not pd.isna(prev['j']) and not pd.isna(day['j']) and day['j'] < prev['j']:
                            sell_count = state['units_held']
                        
                        return {'buy': buy_count, 'sell': sell_count, 'reason': 'kdj_volume'}
                    
                    elif strategy_name == "威廉指标 WR（14）":
                        buy_count = 0
                        sell_count = 0
                        if prev is not None and not pd.isna(prev['wr']) and not pd.isna(day['wr']):
                            if prev['wr'] < -80 and day['wr'] >= -80:
                                buy_count = 1
                            if prev['wr'] < -20 and day['wr'] >= -20:
                                buy_count = min(remaining_units, max(buy_count, 2))
                            if prev['wr'] > -20 and day['wr'] < -20 and day['wr'] < prev['wr']:
                                sell_count = state['units_held']
                        
                        buy_count = min(buy_count, remaining_units)
                        sell_count = min(sell_count, state['units_held'])
                        return {'buy': buy_count, 'sell': sell_count, 'reason': 'wr'}
                    
                    elif strategy_name == "DMI 趋势确认突破":
                        buy_count = 0
                        sell_count = 0
                        if prev is not None and not pd.isna(day['plus_di']) and not pd.isna(day['minus_di']) and not pd.isna(day['adx']) and not pd.isna(prev['adx']):
                            if prev['plus_di'] <= prev['minus_di'] and day['plus_di'] > day['minus_di'] and day['adx'] > 25 and day['adx'] > prev['adx']:
                                buy_count = 2 if remaining_units >= 2 else (1 if remaining_units >= 1 else 0)
                            if prev['minus_di'] <= prev['plus_di'] and day['minus_di'] > day['plus_di']:
                                sell_count = state['units_held']
                            if day['adx'] > 50 and day['adx'] < prev['adx']:
                                sell_count = state['units_held']
                        
                        return {'buy': buy_count, 'sell': sell_count, 'reason': 'dmi'}
                    
                    return {'buy': 0, 'sell': 0, 'reason': 'none'}
                
                # 回测逻辑
                cash = initial_capital
                unit_value = initial_capital / UNIT_COUNT
                units_held = 0
                unit_records = []
                holding_code = None
                trade_count = 0
                trade_log = []
                
                # 用于绘图的数据
                asset_dates = []
                asset_values = []
                return_values = []
                
                # 按日期顺序处理
                all_dates = sorted(df['日期'].unique())
                
                for current_date in all_dates:
                    # 计算当日持仓市值
                    market_value = 0
                    if units_held > 0 and holding_code:
                        code_data = data_by_code[holding_code]
                        day_data = code_data[code_data['日期'] == current_date]
                        if not day_data.empty:
                            price = day_data.iloc[0]['收盘价']
                            market_value = sum(u['shares'] * price for u in unit_records)
                        else:
                            # 使用最后可用价格
                            last_data = code_data[code_data['日期'] <= current_date].tail(1)
                            if not last_data.empty:
                                price = last_data.iloc[0]['收盘价']
                                market_value = sum(u['shares'] * price for u in unit_records)
                    
                    total_asset = cash + market_value
                    asset_dates.append(current_date)
                    asset_values.append(total_asset)
                    return_values.append((total_asset - initial_capital) / initial_capital * 100)
                    
                    # 每日逻辑
                    if units_held == 0:
                        # 遍历股票寻找买入信号
                        for code in data_by_code:
                            code_data = data_by_code[code]
                            day_data = code_data[code_data['日期'] == current_date]
                            if not day_data.empty:
                                day_index = day_data.index[0]
                                state = {'units_held': units_held, 'unit_records': unit_records, 'unit_value': unit_value, 'cash': cash}
                                decision = strategy_decision(strategy, code_data, day_index, state)
                                
                                if decision['buy'] > 0:
                                    to_buy = min(decision['buy'], UNIT_COUNT - units_held)
                                    bought_any = False
                                    
                                    for _ in range(to_buy):
                                        price = day_data.iloc[0]['收盘价']
                                        max_shares = int(unit_value / (price * (1 + FEE_RATE)))
                                        if max_shares <= 0:
                                            break
                                        
                                        cost = max_shares * price
                                        fee = cost * FEE_RATE
                                        
                                        if cash >= cost + fee:
                                            cash -= (cost + fee)
                                            tag = 'bottom' if decision['reason'] == 'bollinger_kdj' and day_data.iloc[0]['收盘价'] < day_data.iloc[0]['bb_lower'] else 'normal'
                                            unit_records.append({'shares': max_shares, 'price': price, 'date': current_date, 'tag': tag})
                                            units_held += 1
                                            trade_count += 1
                                            trade_log.append({
                                                '日期': current_date,
                                                '股票代码': code,
                                                '操作': '买入',
                                                '价格': price,
                                                '份数': 1,
                                                '当次买入股数': max_shares,
                                                '手续费': fee,
                                                '资金余额': cash
                                            })
                                            bought_any = True
                                            holding_code = code
                                        else:
                                            break
                                    
                                    if bought_any:
                                        break
                    else:
                        # 已有持仓，检查卖出信号
                        if holding_code in data_by_code:
                            code_data = data_by_code[holding_code]
                            day_data = code_data[code_data['日期'] == current_date]
                            
                            if not day_data.empty:
                                day_index = day_data.index[0]
                                state = {'units_held': units_held, 'unit_records': unit_records, 'unit_value': unit_value, 'cash': cash}
                                decision = strategy_decision(strategy, code_data, day_index, state)
                                
                                # 执行卖出
                                if decision['sell'] > 0:
                                    sell_count = min(decision['sell'], units_held)
                                    sell_indices = []
                                    
                                    # 优先卖出bottom标签的份额
                                    for i in range(len(unit_records)-1, -1, -1):
                                        if len(sell_indices) >= sell_count:
                                            break
                                        if unit_records[i]['tag'] == 'bottom':
                                            sell_indices.append(i)
                                    
                                    # 若还需卖出，按后进先出
                                    for i in range(len(unit_records)-1, -1, -1):
                                        if len(sell_indices) >= sell_count:
                                            break
                                        if i not in sell_indices:
                                            sell_indices.append(i)
                                    
                                    # 执行卖出操作
                                    for idx in sorted(sell_indices, reverse=True):
                                        rec = unit_records[idx]
                                        price = day_data.iloc[0]['收盘价']
                                        revenue = rec['shares'] * price
                                        fee = revenue * FEE_RATE
                                        cash += revenue - fee
                                        trade_count += 1
                                        trade_log.append({
                                            '日期': current_date,
                                            '股票代码': holding_code,
                                            '操作': '卖出',
                                            '价格': price,
                                            '份数': 1,
                                            '当次买入股数': 0,
                                            '手续费': fee,
                                            '资金余额': cash
                                        })
                                        del unit_records[idx]
                                    
                                    units_held = len(unit_records)
                                    if units_held == 0:
                                        holding_code = None
                                
                                # 执行加仓
                                if decision['buy'] > 0:
                                    to_buy = min(decision['buy'], UNIT_COUNT - units_held)
                                    for _ in range(to_buy):
                                        price = day_data.iloc[0]['收盘价']
                                        max_shares = int(unit_value / (price * (1 + FEE_RATE)))
                                        if max_shares <= 0:
                                            break
                                        
                                        cost = max_shares * price
                                        fee = cost * FEE_RATE
                                        
                                        if cash >= cost + fee:
                                            cash -= (cost + fee)
                                            tag = 'bottom' if decision['reason'] == 'bollinger_kdj' and day_data.iloc[0]['收盘价'] < day_data.iloc[0]['bb_lower'] else 'normal'
                                            unit_records.append({'shares': max_shares, 'price': price, 'date': current_date, 'tag': tag})
                                            units_held += 1
                                            trade_count += 1
                                            trade_log.append({
                                                '日期': current_date,
                                                '股票代码': holding_code,
                                                '操作': '加仓',
                                                '价格': price,
                                                '份数': 1,
                                                '当次买入股数': max_shares,
                                                '手续费': fee,
                                                '资金余额': cash
                                            })
                                        else:
                                            break
                
                # 回测结束：平仓所有持仓
                if units_held > 0 and holding_code:
                    code_data = data_by_code[holding_code]
                    last_day = code_data.iloc[-1]
                    last_date = last_day['日期']
                    
                    for rec in unit_records:
                        price = last_day['收盘价']
                        revenue = rec['shares'] * price
                        fee = revenue * FEE_RATE
                        cash += revenue - fee
                        trade_count += 1
                        trade_log.append({
                            '日期': last_date,
                            '股票代码': holding_code,
                            '操作': '平仓',
                            '价格': price,
                            '份数': 1,
                            '当次买入股数': 0,
                            '手续费': fee,
                            '资金余额': cash
                        })
                    
                    units_held = 0
                    unit_records = []
                    holding_code = None
                
                # 计算最终结果
                final_capital = cash
                total_return = (final_capital - initial_capital) / initial_capital * 100
                
                # 显示回测结果
                st.header("回测结果")
                
                # 结果卡片
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("最终资金", f"¥{final_capital:.2f}")
                with col2:
                    st.metric("总收益率", f"{total_return:.2f}%", f"{total_return:.2f}%")
                with col3:
                    st.metric("交易次数", trade_count)
                
                # 图表
                st.subheader("资产和收益曲线")
                chart_data = pd.DataFrame({
                    '日期': asset_dates,
                    '总资产': asset_values,
                    '累计收益率(%)': return_values
                })
                
                # 总资产图表
                fig1 = px.line(chart_data, x='日期', y='总资产', title='总资产变化')
                st.plotly_chart(fig1, use_container_width=True)
                
                # 收益率图表
                fig2 = px.line(chart_data, x='日期', y='累计收益率(%)', title='累计收益率变化')
                st.plotly_chart(fig2, use_container_width=True)
                
                # 交易日志
                st.subheader("交易日志")
                if trade_log:
                    log_df = pd.DataFrame(trade_log)
                    st.dataframe(log_df)
                else:
                    st.info("无交易记录")
                    
        else:
            st.error("CSV文件格式错误，缺少必要的列。请确保文件包含：日期、股票代码、开盘价、收盘价、最高价、最低价、成交量")
            
    except Exception as e:
        st.error(f"读取文件时出错: {e}")
else:
    st.info("请上传CSV格式的股票数据文件")

# 页脚
st.markdown("---")
st.markdown("股票回测工具（Streamlit版） | 支持多种技术分析策略")
