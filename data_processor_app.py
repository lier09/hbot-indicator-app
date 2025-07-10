"""
高压氧/氧气研究 – 指标处理器
=============================================
* **双模式**：Streamlit 网页 / 命令行
* 通过 `st.runtime.exists()` 精准判断是否在 `streamlit run` 环境，
  避免网页模式下再次执行 CLI 入口。
* 未传文件时仅打印帮助并返回，不抛 `SystemExit`。

使用方法
---------
### 网页模式
```bash
pip install streamlit pandas numpy openpyxl
streamlit run data_processor_app.py
```
浏览器访问自动弹出的地址（如 `http://localhost:8501`）。

### 命令行模式
```bash
python data_processor_app.py A.xlsx B.xlsx -o result.tsv
```
未传入文件将打印帮助。
"""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# -------------------------------------------------------------
# 检测是否在 Streamlit 环境 (兼容新版 Streamlit 1.30+)
# -------------------------------------------------------------
try:
    import streamlit as st  # type: ignore

    HAS_STREAMLIT = True
    # 新版 Streamlit 提供 st.runtime.exists() 判断脚本是否
    # 正在 `streamlit run` 的服务器线程中。
    try:
        IS_ST_RUN = st.runtime.exists()
    except AttributeError:
        # 旧版本 fallback：若启动命令包含 "streamlit" 关键词
        IS_ST_RUN = "streamlit" in sys.argv[0]
except ModuleNotFoundError:  # pragma: no cover
    HAS_STREAMLIT = False
    IS_ST_RUN = False

RULE_CV_STABLE = 5  # %


# -------------------------------------------------------------
# 核心计算 (已按要求修改)
# -------------------------------------------------------------

def process_file(df: pd.DataFrame) -> Dict[str, str]:
    """
    处理单个文件：
    1. 剔除前 20 秒适应期 (按 4s/点, 即前 5 个数据点)。
    2. 对剩余数据，取末尾 6 点作为稳态段进行计算。
    3. 对稳态段数据，若某指标 CV > 5%，则尝试用 IQR 方法剔除异常值后重新计算。
    4. 返回处理后的指标字典。
    """
    df = df.copy()
    # 将所有列尽可能转换为数值类型，无法转换的设为 NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. 剔除前 20 秒适应期 (假设 4s/点, 即前 5 个点)
    if len(df) > 5:
        df = df.iloc[5:].reset_index(drop=True)

    # 如果数据不足，直接返回空字典
    if df.empty:
        return {}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    sbp = next((c for c in num_cols if re.fullmatch(r".*SBP.*", c, re.I)), None)
    dbp = next((c for c in num_cols if re.fullmatch(r".*DBP.*", c, re.I)), None)
    mapc = next((c for c in num_cols if re.fullmatch(r".*MAP.*", c, re.I)), None)

    out: Dict[str, str] = {}
    # SBP/DBP/MAP 取适应期后第一个非零值作为基线
    for label, col in [("SBP", sbp), ("DBP", dbp), ("MAP", mapc)]:
        if col is not None:
            # 寻找第一个有效 (非 NaN 且 > 0) 的值
            nz = df.loc[df[col].notna() & (df[col] > 0), col]
            if not nz.empty:
                out[label] = str(int(round(nz.iloc[0])))

    # 2. 稳态段定义为最后 6 个数据点
    stable = df.tail(6)

    for col in num_cols:
        # 跳过已处理的列
        if col in {sbp, dbp, mapc}:
            continue

        vals = stable[col].dropna()
        if vals.empty or (vals == 0).all():
            continue

        # 初始计算均值、标准差、CV
        mean = vals.mean()
        if np.isnan(mean):
            continue
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        # 避免除以零
        cv = abs(sd / mean) * 100 if mean != 0 else 0.0

        # 3. 如果 CV 大于阈值，并且有足够数据进行异常值判断，则处理异常值
        if cv > RULE_CV_STABLE and len(vals) >= 4:
            Q1 = vals.quantile(0.25)
            Q3 = vals.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            vals_cleaned = vals[(vals >= lower_bound) & (vals <= upper_bound)]

            # 如果成功移除了异常值，则使用新数据重新计算
            if not vals_cleaned.empty and len(vals_cleaned) < len(vals):
                mean = vals_cleaned.mean()
                sd = vals_cleaned.std(ddof=1) if len(vals_cleaned) > 1 else 0.0
                cv = abs(sd / mean) * 100 if mean != 0 else 0.0
                # 输出结果中明确标注进行了异常值处理
                out[col] = f"{mean:.2f} (CV>5% 异常值已移除, 新CV {cv:.1f}%)"
            else:
                # 清洗无效或未移除任何值，保留原始高 CV 标记
                out[col] = f"{mean:.2f} (CV {cv:.1f}%)"
        elif cv > RULE_CV_STABLE:
            # 数据点太少无法去异常值，或CV本身就高，直接标记
            out[col] = f"{mean:.2f} (CV {cv:.1f}%)"
        else:
            # CV 稳定，直接输出均值
            out[col] = f"{mean:.2f}"

    return out


def build_wide(summaries: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    """将多个摘要字典合并成一个宽格式的 DataFrame。"""
    all_cols = sorted({c for d in summaries.values() for c in d})
    data = [[subj] + [summaries[subj].get(c, "") for c in all_cols] for subj in sorted(summaries)]
    return pd.DataFrame(data, columns=["受试者"] + all_cols)


# -------------------------------------------------------------
# Streamlit 前端
# -------------------------------------------------------------
if HAS_STREAMLIT and IS_ST_RUN:  # 仅在 `streamlit run` 时执行此代码块
    st.set_page_config(page_title="指标处理器", layout="wide")
    st.title("⚙️ 指标自动处理工具 — HBOT 研究")
    st.markdown("""
    本工具根据以下规则处理数据：
    1.  **自动剔除适应期**：原始数据的前 20 秒（5个数据点）将被剔除。
    2.  **稳态段分析**：使用剩余数据的**最后 30 秒内（末 6 个点）** 作为稳态段进行计算。
    3.  **智能稳定判断与处理**：
        - 计算各指标在稳态段的均值。
        - 若变异系数 **CV ≤ 5%**，则认为数据稳定，直接采用均值。
        - 若 **CV > 5%**，工具将自动**剔除异常值**后重新计算均值，并标记 `(CV>5% 异常值已移除, 新CV X.X%)` 以供参考。
    """)
    uploads = st.file_uploader(
        "上传 4 s/点采样 Excel 原始记录（可多选）",
        type=["xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploads:
        summaries = {}
        for up in uploads:
            try:
                # 从文件名提取受试者名称 (例如 "S01_data.xlsx" -> "S01")
                subject_name = up.name.split("_")[0]
                summaries[subject_name] = process_file(pd.read_excel(up, engine='openpyxl'))
            except Exception as exc:
                st.error(f"读取失败 {up.name}: {exc}")

        if summaries:
            wide_df = build_wide(summaries)
            st.dataframe(wide_df, use_container_width=True)
            st.download_button(
                label="⬇️ 下载 TSV",
                data=wide_df.to_csv(sep="\t", index=False).encode("utf-8"),
                file_name="合并结果.tsv",
                mime="text/tab-separated-values",
            )
    # 在现代 Streamlit 应用中不再需要 st.stop()


# -------------------------------------------------------------
# CLI (命令行界面) 模式
# -------------------------------------------------------------

def cli(argv: List[str] | None = None) -> None:  # pragma: no cover
    """处理命令行执行的函数。"""
    parser = argparse.ArgumentParser(
        description="""
        指标处理器 (命令行模式)。
        处理 Excel 文件：首先移除 20 秒的适应期，
        然后分析最后 6 个数据点作为稳态段。
        如果稳态段的 CV > 5%，则在计算均值前移除异常值。
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("files", nargs="*", help="需要处理的 Excel 文件 (.xlsx, .xls)")
    parser.add_argument("-o", "--output", default="合并指标.tsv", help="输出的 TSV 文件名")

    # 使用 parse_known_args 忽略由执行环境传入的未知参数
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        print(f"[警告] 忽略无法识别的参数: {' '.join(unknown)}")

    if not args.files:
        parser.print_help()
        return  # 返回而不是退出，便于交互式调用

    summaries: Dict[str, Dict[str, str]] = {}
    for fp in args.files:
        p = Path(fp)
        # **修正**: 仅处理有效的 Excel 文件后缀，避免读取流。
        if p.suffix.lower() not in ['.xlsx', '.xls']:
            print(f"[信息] 跳过非 Excel 文件: {p}")
            continue
        try:
            # 从文件名主干提取受试者名称
            subject_name = p.stem.split("_")[0]
            summaries[subject_name] = process_file(pd.read_excel(p, engine='openpyxl'))
        except FileNotFoundError:
            print(f"[警告] 找不到文件: {p}")
        except Exception as exc:
            print(f"[警告] 读取 {p.name} 失败: {exc}")

    if not summaries:
        print("⚠️ 未能处理任何数据。")
        return

    wide_df = build_wide(summaries)
    wide_df.to_csv(args.output, sep="\t", index=False, encoding='utf-8')
    print(f"✅ TSV 文件已保存 -> {args.output}")
    print("\n预览 (前 5 行):\n", wide_df.head().to_string(index=False))


if __name__ == "__main__":
    # 主入口点确保 CLI 函数仅在直接执行脚本时被调用，
    # 而不是在由 Streamlit 运行时被调用。
    if not IS_ST_RUN:
        cli(sys.argv[1:])
