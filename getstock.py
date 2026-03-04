import akshare as ak
import pandas as pd
from datetime import datetime, timedelta


def download_a_stock_data(stock_code: str):
    """
    下载A股最近一年数据并保存为CSV
    """

    # ===== 计算日期范围 =====
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)

    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    print(f"正在下载 {stock_code} 从 {start_str} 到 {end_str} 的数据...")

    # ===== 获取股票数据 =====
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_str,
        end_date=end_str,
        adjust=""
    )

    if df.empty:
        print("未获取到数据，请检查股票代码")
        return

    # ===== 选择并重命名字段 =====
    df = df[["日期", "开盘", "收盘", "最高", "最低", "成交量"]]

    df.rename(columns={
        "开盘": "开盘价",
        "收盘": "收盘价",
        "最高": "最高价",
        "最低": "最低价"
    }, inplace=True)

    # 添加股票代码列
    df["股票代码"] = stock_code

    # 调整列顺序
    df = df[
        ["日期", "股票代码", "开盘价", "收盘价", "最高价", "最低价", "成交量"]
    ]

    # ===== 保存CSV =====
    filename = f"{stock_code}_last_year.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")

    print(f"✅ 数据已保存为: {filename}")


# ===== 用户输入 =====
if __name__ == "__main__":
    code = input("请输入A股股票代码（如 600519）：").strip()
    download_a_stock_data(code)