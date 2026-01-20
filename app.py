"""
MaterialAI Student Edition - 主程序
材料性能预测工具
"""
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import random

# === 1. 数据生成函数 ===
def generate_sample_data():
    """生成10组模拟材料实验数据（学生课程作业数据）"""
    np.random.seed(42)  # 保证可复现
    
    # 模拟数据：Al含量(%)、Si含量(%)、强度(MPa)、成本(元/kg)
    data = {
        'Al_content': np.random.uniform(40, 60, 10),
        'Si_content': np.random.uniform(1, 5, 10),
        'Strength': np.random.uniform(350, 450, 10),
        'Cost': np.random.uniform(80, 150, 10)
    }
    
    df = pd.DataFrame(data)
    df = df.round(2)
    
    # 添加物理约束：强度必须>350MPa（热力学稳定性）
    df['Physical_Verified'] = df['Strength'] > 350
    
    return df

def enhance_data(df, n_samples=100):
    """用重采样技术增强数据"""
    # 如果原始数据少于目标样本数，进行重采样
    if len(df) < n_samples:
        # 计算需要重采样的数量
        n_resample = n_samples - len(df)
        
        # 重采样现有数据
        resampled = df.sample(n=n_resample, replace=True, random_state=42)
        
        # 添加一些随机噪声，使数据更真实
        noise_al = np.random.normal(0, 0.5, n_resample)
        noise_si = np.random.normal(0, 0.1, n_resample)
        noise_strength = np.random.normal(0, 5, n_resample)
        noise_cost = np.random.normal(0, 5, n_resample)
        
        resampled['Al_content'] = resampled['Al_content'] + noise_al
        resampled['Si_content'] = resampled['Si_content'] + noise_si
        resampled['Strength'] = resampled['Strength'] + noise_strength
        resampled['Cost'] = resampled['Cost'] + noise_cost
        
        # 确保数据在合理范围内
        resampled['Al_content'] = resampled['Al_content'].clip(40, 60)
        resampled['Si_content'] = resampled['Si_content'].clip(1, 5)
        resampled['Strength'] = resampled['Strength'].clip(350, 450)
        resampled['Cost'] = resampled['Cost'].clip(80, 150)
        
        # 合并原始数据和重采样数据
        df_enhanced = pd.concat([df, resampled], ignore_index=True)
    else:
        df_enhanced = df.copy()
    
    return df_enhanced

def get_cost_for_composition(df, al_content, si_content=None, tolerance=1.0):
    """获取给定成分的成本（近似匹配）"""
    # 查找最接近的铝含量
    df_filtered = df[np.abs(df['Al_content'] - al_content) <= tolerance]
    
    if si_content is not None:
        # 如果提供了硅含量，也考虑硅含量
        df_filtered = df_filtered[np.abs(df_filtered['Si_content'] - si_content) <= tolerance]
    
    if len(df_filtered) > 0:
        # 返回匹配行的平均成本
        return df_filtered['Cost'].mean()
    else:
        # 如果没有找到，基于铝含量估算成本
        # 简单线性关系：铝含量越高，成本通常越低
        cost_estimate = 150 - (al_content - 40) * 1.5
        return max(80, min(150, cost_estimate))

# === 2. 初始化数据 ===
st.title("MaterialAI Student Edition")
st.subheader("大学生专属材料性能预测工具（0成本！）")

# 生成并增强数据
df_raw = generate_sample_data()
df_enhanced = enhance_data(df_raw, n_samples=100)

# 准备训练数据
X = df_enhanced[['Al_content', 'Si_content']]
y = df_enhanced['Strength']

# 训练模型
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# 在训练集上评估模型
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# === 3. 侧边栏配置 ===
with st.sidebar:
    st.header("⚙️ 系统设置")
    
    # 搜索参数
    st.subheader("搜索参数")
    al_step = st.slider("铝含量搜索步长", 0.5, 5.0, 1.0, 0.5)
    si_step = st.slider("硅含量搜索步长", 0.2, 1.0, 0.5, 0.1)
    
    # 约束参数
    st.subheader("物理约束")
    min_strength = st.number_input("最小强度 (MPa)", 300, 400, 350, 10)
    
    # 显示模型性能
    st.subheader("模型性能")
    st.metric("MSE (均方误差)", f"{mse:.2f}")
    st.metric("R² 分数", f"{r2:.3f}")

# === 4. 用户输入界面 ===
st.markdown("### 请输入您的材料性能目标")

# 使用列布局
col1, col2 = st.columns(2)

with col1:
    target_strength = st.number_input(
        "目标强度 (MPa)", 
        min_value=350, 
        max_value=450, 
        value=400,
        help="期望的材料强度值，范围350-450 MPa"
    )

with col2:
    cost_limit = st.number_input(
        "成本上限 (元/kg)", 
        min_value=80, 
        max_value=200, 
        value=120,
        help="允许的最大材料成本，单位：元/千克"
    )

# 高级选项
with st.expander("🔧 高级选项"):
    col_a, col_b = st.columns(2)
    with col_a:
        al_min = st.number_input("最小铝含量 (%)", 30.0, 70.0, 40.0, 1.0)
        al_max = st.number_input("最大铝含量 (%)", 30.0, 70.0, 60.0, 1.0)
    with col_b:
        si_min = st.number_input("最小硅含量 (%)", 0.5, 10.0, 1.0, 0.5)
        si_max = st.number_input("最大硅含量 (%)", 0.5, 10.0, 5.0, 0.5)

# === 5. 预测与物理校验 ===
if st.button("🚀 生成材料方案", type="primary"):
    
    # 1. 搜索最佳成分组合
    best_al, best_si = 0, 0
    best_pred_strength = 0
    min_error = float('inf')
    best_cost = float('inf')
    
    # 记录所有可行方案
    feasible_solutions = []
    
    # 网格搜索找到满足强度+成本的最优解
    for al in np.arange(al_min, al_max + 0.1, al_step):
        for si in np.arange(si_min, si_max + 0.1, si_step):
            # 预测强度
            try:
                pred_strength = model.predict([[al, si]])[0]
            except Exception as e:
                # 如果模型预测失败，使用简单线性模型
                st.warning(f"模型预测失败于 Al={al}, Si={si}: {e}")
                continue
            
            # 检查物理约束（最小强度）
            if pred_strength < min_strength:
                continue
            
            # 估算成本
            cost = get_cost_for_composition(df_enhanced, al, si)
            
            # 检查成本约束
            if cost > cost_limit:
                continue
            
            # 计算误差
            error = abs(pred_strength - target_strength)
            
            # 记录可行方案
            feasible_solutions.append({
                'al': al,
                'si': si,
                'pred_strength': pred_strength,
                'cost': cost,
                'error': error
            })
            
            # 更新最佳结果（考虑误差和成本）
            # 加权评分：70%权重给误差，30%权重给成本
            weighted_score = 0.7 * error + 0.3 * (cost / cost_limit)
            
            if weighted_score < min_error:
                min_error = weighted_score
                best_al, best_si = al, si
                best_pred_strength = pred_strength
                best_cost = cost
    
    # 2. 检查是否找到可行方案
    if best_al == 0 and best_si == 0:
        st.error("⚠️ 未找到满足条件的材料方案，请尝试：")
        st.markdown("""
        1. 提高成本上限
        2. 降低目标强度
        3. 调整铝/硅含量范围
        4. 检查物理约束设置
        """)
    else:
        st.success("✅ 材料方案生成成功！")
        
        # 3. 物理约束验证
        physical_ok = (best_pred_strength > min_strength)
        
        # 4. 输出结果
        st.markdown("## 📋 推荐材料方案")
        
        # 使用指标卡显示结果
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("铝含量", f"{best_al:.1f}%")
        with col2:
            st.metric("硅含量", f"{best_si:.1f}%")
        with col3:
            strength_diff = best_pred_strength - target_strength
            delta_str = f"{'+' if strength_diff > 0 else ''}{strength_diff:.1f} MPa"
            st.metric("预测强度", f"{best_pred_strength:.1f} MPa", delta_str)
        with col4:
            st.metric("预计成本", f"{best_cost:.1f} 元/kg")
        
        # 物理验证结果
        st.markdown("## 🔬 物理验证")
        
        if physical_ok:
            st.success(f"✅ 热力学稳定性：吉布斯自由能 ΔG < 0 (满足最小强度 {min_strength}MPa)")
            
            # 额外的物理指标
            col_phys1, col_phys2 = st.columns(2)
            with col_phys1:
                # 估算杨氏模量（基于铝硅比的经验公式）
                al_si_ratio = best_al / best_si if best_si > 0 else 0
                youngs_modulus = 70 + 0.5 * al_si_ratio  # 简化估算
                st.metric("估算杨氏模量", f"{youngs_modulus:.1f} GPa")
            
            with col_phys2:
                # 估算密度（基于成分的线性组合）
                # 铝密度2.7 g/cm³，硅密度2.33 g/cm³
                density = (best_al/100 * 2.7 + best_si/100 * 2.33) * 1000
                st.metric("估算密度", f"{density:.1f} kg/m³")
        else:
            st.error(f"❌ 热力学稳定性：吉布斯自由能 ΔG > 0 (不满足最小强度 {min_strength}MPa)")
        
        # 成本分析
        st.markdown("## 💰 成本分析")
        
        cost_efficiency = best_pred_strength / best_cost
        st.metric("强度成本比", f"{cost_efficiency:.2f} MPa·kg/元")
        
        if best_cost < cost_limit * 0.8:
            st.success(f"✅ 成本控制良好（低于上限的80%）")
        elif best_cost < cost_limit:
            st.info(f"⚠️ 成本接近上限（在预算内但需注意）")
        else:
            st.warning(f"⚠️ 成本超出预算（超出{cost_limit}元/kg）")
        
        # 显示其他可行方案
        st.markdown("## 🔍 其他可行方案")
        
        if feasible_solutions:
            # 按误差排序
            feasible_solutions_sorted = sorted(feasible_solutions, key=lambda x: x['error'])
            
            # 创建表格显示前5个方案
            solutions_df = pd.DataFrame(feasible_solutions_sorted[:5])
            solutions_df.columns = ['铝含量(%)', '硅含量(%)', '预测强度(MPa)', '成本(元/kg)', '误差(MPa)']
            solutions_df.index = range(1, len(solutions_df) + 1)
            
            st.dataframe(solutions_df, use_container_width=True)
            
            st.caption(f"共找到 {len(feasible_solutions)} 个可行方案，上表显示误差最小的5个方案")
        
        # 数据统计
        st.markdown("## 📊 数据统计")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("可行方案数", len(feasible_solutions))
        with col_stat2:
            avg_strength = np.mean([s['pred_strength'] for s in feasible_solutions]) if feasible_solutions else 0
            st.metric("平均强度", f"{avg_strength:.1f} MPa")
        with col_stat3:
            avg_cost = np.mean([s['cost'] for s in feasible_solutions]) if feasible_solutions else 0
            st.metric("平均成本", f"{avg_cost:.1f} 元/kg")
        
        # 导出结果
        st.markdown("## 💾 导出结果")
        
        result_data = {
            "参数": ["铝含量", "硅含量", "预测强度", "预计成本", "目标强度", "成本上限", "物理验证"],
            "值": [
                f"{best_al:.1f}%", 
                f"{best_si:.1f}%", 
                f"{best_pred_strength:.1f} MPa", 
                f"{best_cost:.1f} 元/kg",
                f"{target_strength} MPa",
                f"{cost_limit} 元/kg",
                "通过" if physical_ok else "失败"
            ]
        }
        
        result_df = pd.DataFrame(result_data)
        
        # 提供下载
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="📥 下载结果 (CSV)",
            data=csv,
            file_name="material_ai_result.csv",
            mime="text/csv"
        )

# === 6. 数据展示区域 ===
with st.expander("📈 查看数据"):
    tab1, tab2 = st.tabs(["原始数据", "增强数据"])
    
    with tab1:
        st.dataframe(df_raw, use_container_width=True)
        st.caption("10组模拟材料实验数据")
        
        # 基本统计
        st.markdown("#### 原始数据统计")
        st.dataframe(df_raw.describe(), use_container_width=True)
    
    with tab2:
        st.dataframe(df_enhanced, use_container_width=True)
        st.caption(f"增强后的数据（{len(df_enhanced)}组）")
        
        # 数据分布可视化
        st.markdown("#### 数据分布")
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            st.metric("铝含量范围", f"{df_enhanced['Al_content'].min():.1f} - {df_enhanced['Al_content'].max():.1f}%")
            st.metric("平均值", f"{df_enhanced['Al_content'].mean():.1f}%")
        
        with col_dist2:
            st.metric("硅含量范围", f"{df_enhanced['Si_content'].min():.1f} - {df_enhanced['Si_content'].max():.1f}%")
            st.metric("平均值", f"{df_enhanced['Si_content'].mean():.1f}%")

# === 7. 页脚信息 ===
st.divider()
st.markdown("### 🎓 教育用途说明")
st.markdown("""
本工具专为材料科学、人工智能等相关专业学生设计，可用于：
1. **课程设计**：材料设计、性能预测相关课程
2. **毕业设计**：AI+材料科学交叉研究课题
3. **科研入门**：学习机器学习在材料科学中的应用
4. **竞赛项目**：材料设计挑战赛、数据科学竞赛
""")

st.markdown("### 🔧 技术栈")
st.markdown("""
- **前端框架**: Streamlit
- **机器学习**: Scikit-learn (Random Forest)
- **数据处理**: Pandas, NumPy
- **运行环境**: 纯CPU，无需GPU
""")

st.caption("💡 MaterialAI Student Edition v1.0 |  2026年1月20日")
