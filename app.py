import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. 页面基础设置
st.set_page_config(
    page_title="Gastric Cancer Metastasis Calculator",
    page_icon="🩺",
    layout="wide"
)


# 2. 加载训练好的模型与特征列结构
@st.cache_resource
def load_assets():
    model = joblib.load("best_model.pkl")
    columns = joblib.load("model_columns.pkl")
    return model, columns


model, model_columns = load_assets()

st.title("Gastric Cancer Distant Metastasis Risk Calculator")
st.markdown("基于最优机器学习模型（XGBoost）开发，用于评估胃癌患者远处转移（DM）的个体化风险。")
st.write("---")

# 3. 临床特征输入表单
st.subheader("📋 请输入患者临床病理特征")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.selectbox("年龄 (Age)", options=["≤50", ">50"], index=0)
    sex = st.selectbox("性别 (Sex)", options=["Male", "Female"], index=0)
    race = st.selectbox("种族 (Race)", options=["White", "Black", "Other"], index=0)

with col2:
    t_stage = st.selectbox("T分期 (T stage)", options=["T1", "T2", "T3", "T4", "Tx"], index=0)
    n_stage = st.selectbox("N分期 (N stage)", options=["N0", "N1", "N2", "N3", "Nx"], index=0)
    grade = st.selectbox("病理分级 (Grade)", options=["Grade I", "Grade II", "Grade III", "Grade IV"], index=0)

with col3:
    tumor_size = st.selectbox("肿瘤大小 (Tumor size)", options=["≤5", ">5"], index=0)
    chemotherapy = st.selectbox("化疗 (Chemotherapy)", options=["No", "Yes"], index=0)
    radiation = st.selectbox("放疗 (Radiation)", options=["No", "Yes"], index=0)

st.write("---")

# 4. 计算与预测
if st.button("开始预测远处转移风险", type="primary", use_container_width=True):
    # 构建基准全 0 特征行，列名严格与模型完全一致
    input_aligned = pd.DataFrame(0, index=[0], columns=model_columns)

    # 收集当前用户选中的所有特征标签
    selected_features = [
        f"Age_{age}",
        f"Sex_{sex}",
        f"Race_{race}",
        f"T_stage_{t_stage}",
        f"N_stage_{n_stage}",
        f"Grade_{grade}",
        f"Tumor_size_{tumor_size}",
        f"Chemotherapy_{chemotherapy}",
        f"Radiation_{radiation}"
    ]

    # 遍历选中的特征，只要该特征列存在于模型中（非 Ref 对照组），就置为 1
    activated = []
    for feat in selected_features:
        if feat in input_aligned.columns:
            input_aligned.at[0, feat] = 1
            activated.append(feat)

    # 预测阳性概率
    prob = model.predict_proba(input_aligned)[0, 1]
    risk_pct = prob * 100

    # 5. 展示预测结果
    st.subheader("🎯 预测结果")
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="远处转移预测概率 (DM Risk)", value=f"{risk_pct:.2f}%")
        st.caption(f"当前激活的模型哑变量: {', '.join(activated) if activated else '全部为对照组基线 (Ref)'}")

    with res_col2:
        if risk_pct >= 50.0:
            st.error(f"⚠️ 高危转移风险（{risk_pct:.2f}%）：建议强化术后随访频次及腹腔/全身强化影像学筛查。")
        elif risk_pct >= 20.0:
            st.warning(f"⚡ 中度转移风险（{risk_pct:.2f}%）：建议密切监测肿瘤标志物及常规复查。")
        else:
            st.success(f"✅ 低转移风险（{risk_pct:.2f}%）：当前特征提示远处转移风险处于较低水平。")

    # 展开查看调试数据，确保特征矩阵准确赋值
    with st.expander("🔍 查看送入模型的完整特征矩阵 (Debug)"):
        st.dataframe(input_aligned)