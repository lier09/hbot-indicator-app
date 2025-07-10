"""
Hyperbaric/Oxygen Study – Indicator Processor
=============================================
* **双模式**：Streamlit 网页 / 命令行
* 通过 `st._is_running_with_streamlit` 精准判断是否在 `streamlit run` 环境，
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
# 检测是否在 Streamlit 环境  (兼容新版 Streamlit 1.30+)
# -------------------------------------------------------------
try:
    import streamlit as st  # type: ignore
    HAS_STREAMLIT = True
    # 新版 Streamlit 提供 st.runtime.exists() 判断脚本是否
    # 正在 `streamlit run` 的服务器线程中。
    try:
        IS_ST_RUN = st.runtime.exists()  # type: ignore[attr-defined]
    except Exception:
        # 旧版本 fallback：若启动命令包含 "streamlit" 关键词
        IS_ST_RUN = "streamlit" in sys.argv[0]
except ModuleNotFoundError:  # pragma: no cover
    HAS_STREAMLIT = False
    IS_ST_RUN = False

RULE_CV_STABLE = 5  # %

# -------------------------------------------------------------
# 核心计算
# -------------------------------------------------------------

def process_file(df: pd.DataFrame) -> Dict[str, str]:
    """返回稳态段（末 6 点）统计后的指标字典。"""
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    sbp = next((c for c in num_cols if re.fullmatch(r".*SBP.*", c, re.I)), None)
    dbp = next((c for c in num_cols if re.fullmatch(r".*DBP.*", c, re.I)), None)
    mapc = next((c for c in num_cols if re.fullmatch(r".*MAP.*", c, re.I)), None)

    out: Dict[str, str] = {}
    for label, col in [("SBP", sbp), ("DBP", dbp), ("MAP", mapc)]:
        if col is not None:
            nz = df.loc[df[col] > 0, col]
            if not nz.empty:
                out[label] = str(int(round(nz.iloc[0])))

    stable = df.tail(6)
    for col in num_cols:
        if col in {sbp, dbp, mapc}:
            continue
        vals = stable[col].dropna()
        if vals.empty or (vals == 0).all():
            continue
        mean = vals.mean()
        if mean == 0 or np.isnan(mean):
            continue
        sd = vals.std(ddof=1) if len(vals) > 1 else 0
        cv = abs(sd / mean) * 100
        out[col] = f"{mean:.2f}" if cv <= RULE_CV_STABLE else f"{mean:.2f} (CV {cv:.1f}%)"
    return out


def build_wide(summaries: Dict[str, Dict[str, str]]) -> pd.DataFrame:
    all_cols = sorted({c for d in summaries.values() for c in d})
    data = [[subj] + [summaries[subj].get(c, "") for c in all_cols] for subj in sorted(summaries)]
    return pd.DataFrame(data, columns=["Subject"] + all_cols)

# -------------------------------------------------------------
# Streamlit 前端
# -------------------------------------------------------------
if HAS_STREAMLIT and IS_ST_RUN:  # 仅在真正 `streamlit run` 时进入
    st.set_page_config(page_title="Indicator Processor", layout="wide")
    st.title("⚙️ 指标自动处理工具 — HBOT Study")
    uploads = st.file_uploader("上传 4 s/点采样 Excel 原始记录（可多选）", type="xlsx", accept_multiple_files=True)

    if uploads:
        summaries = {}
        for up in uploads:
            try:
                summaries[up.name.split("_")[0]] = process_file(pd.read_excel(up))
            except Exception as exc:
                st.error(f"读取失败 {up.name}: {exc}")
        if summaries:
            wide_df = build_wide(summaries)
            st.dataframe(wide_df, use_container_width=True)
            st.download_button(
                "⬇️ 下载 TSV",
                wide_df.to_csv(sep="\t", index=False).encode(),
                "combined.tsv",
                "text/tab-separated-values",
            )
    st.stop()

# -------------------------------------------------------------
# CLI 模式
# -------------------------------------------------------------

def cli(argv: List[str] | None = None) -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Indicator Processor (CLI mode)")
    parser.add_argument("files", nargs="*", help="Excel files to process")
    parser.add_argument("-o", "--output", default="combined_indicators.tsv", help="Output TSV filename")
    args = parser.parse_args(argv)

    if not args.files:
        parser.print_help()
        return  # 不退出整个 Python 进程，便于交互式调用

    summaries: Dict[str, Dict[str, str]] = {}
    for fp in args.files:
        p = Path(fp)
        try:
            summaries[p.stem.split("_")[0]] = process_file(pd.read_excel(p))
        except FileNotFoundError:
            print(f"[WARN] 找不到文件: {p}")
        except Exception as exc:
            print(f"[WARN] 读取失败 {p.name}: {exc}")

    if not summaries:
        print("⚠️ 没有可处理的数据。")
        return

    wide_df = build_wide(summaries)
    wide_df.to_csv(args.output, sep="\t", index=False)
    print(f"✅ TSV 已保存 -> {args.output}")
    print("\n预览 (前 5 行):\n", wide_df.head().to_string(index=False))


if __name__ == "__main__":
    cli(sys.argv[1:])
