import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import numpy as np

# ==========================================
# 1. 設定與常數
# ==========================================
GEMINI_MODELS = [
    "models/gemini-flash-latest",
    "models/gemini-2.5-flash",
    "models/gemini-pro-latest",
    "models/gemini-2.5-pro",
]

# 指定分析的重點國家清單
TARGET_COUNTRIES = [
    "Taiwan", "Hong Kong", "Japan", "South Korea", "Thailand", 
    "Vietnam", "Philippines", "Singapore", "China", 
    "United States", "Canada", "United Kingdom", "France", 
    "Sweden", "Norway"
]

st.set_page_config(page_title="Netflix 數據戰情室 V6.2", layout="wide")
st.title("🎬 Netflix 深度數據分析系統")

# ==========================================
# 2. 資料讀取
# ==========================================
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['week'] = pd.to_datetime(df['week'])
        df['Week_Str'] = df['week'].dt.strftime('%Y-%m-%d')
        
        # 確保 Views 相關欄位是數字
        view_cols = [c for c in df.columns if 'Views' in c]
        for col in view_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
        
        return df
    except FileNotFoundError:
        st.error(f"找不到檔案 '{file_path}'")
        return pd.DataFrame()

df_raw = load_data('總表(new)_20251027.zip')

if df_raw.empty:
    st.stop()

# ==========================================
# 3. 側邊欄設定
# ==========================================
st.sidebar.header("⚙️ 參數設定")

category_mode = st.sidebar.radio("內容類別", ("Films", "TV"), index=0)
df_main = df_raw[df_raw['category'] == category_mode].copy()

gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
selected_model = st.sidebar.selectbox("AI 模型", GEMINI_MODELS)

st.sidebar.markdown("---")

# ==========================================
# 4. Gemini Helper
# ==========================================
def ask_gemini(api_key, prompt, model_name):
    if not api_key:
        return "⚠️ 請輸入 API Key"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==========================================
# 5. 分析核心類別
# ==========================================
class NetflixAnalyzerV6:
    def __init__(self, df, api_key, model_name):
        self.df = df.copy()
        self.api_key = api_key
        self.model_name = model_name

    # -------------------------------------------------------------------------
    #  A. 觀看國視角
    # -------------------------------------------------------------------------
    def analyze_viewer(self, target_country):
        st.header(f"🌍 消費市場分析：{target_country} ({category_mode})")
        
        if target_country not in self.df['country_name'].unique():
            st.warning(f"⚠️ 資料庫中沒有 {target_country} 的觀看數據。")
            return

        filtered_df = self.df[self.df['country_name'] == target_country].copy()
        domestic_export_df = self.df[self.df['Country'] == target_country].copy()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 來源排名(量)", "🏆 冠軍來源國", "🗺️ 來源地圖", "🚀 本國輸出表現",
            "🔥 熱門作品", "📑 詳細清單", "💾 原始數據"
        ])

        with tab1:
            unique_counts = filtered_df.groupby('Country')['show_title'].nunique().reset_index(name='Unique_Titles').sort_values('Unique_Titles', ascending=False)
            fig = px.bar(unique_counts, x='Unique_Titles', y='Country', orientation='h', text_auto=True, title=f"{target_country} 的內容供應國排名 (依片量)", color='Unique_Titles', color_continuous_scale='Viridis')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(unique_counts, use_container_width=True)

        with tab2:
            rank1_df = filtered_df[filtered_df['weekly_rank'] == 1]
            if rank1_df.empty: st.info("無冠軍數據")
            else:
                rank1_counts = rank1_df['Country'].value_counts().reset_index()
                rank1_counts.columns = ['Producer_Country', 'Weeks_at_No1']
                c1, c2 = st.columns([1, 1])
                with c1: st.plotly_chart(px.pie(rank1_counts, values='Weeks_at_No1', names='Producer_Country', title='冠軍週數佔比'), use_container_width=True)
                with c2: st.dataframe(rank1_counts, use_container_width=True)
                st.dataframe(rank1_df.groupby('Country')['show_title'].unique().apply(lambda x: ", ".join(x)).reset_index(name='Champion_Titles'), use_container_width=True)

        with tab3:
            unique_counts = filtered_df.groupby('Country')['show_title'].nunique().reset_index(name='Unique_Titles') 
            st.plotly_chart(px.choropleth(unique_counts, locations="Country", locationmode="country names", color="Unique_Titles", color_continuous_scale='Greens', title=f"{target_country} 的內容進口地圖"), use_container_width=True)

        with tab4:
            if domestic_export_df.empty: st.warning("無自製內容數據")
            else:
                export_stats = domestic_export_df.groupby('country_name')['show_title'].nunique().reset_index(name='Titles_Count').sort_values('Titles_Count', ascending=False)
                st.plotly_chart(px.choropleth(export_stats, locations="country_name", locationmode="country names", color="Titles_Count", color_continuous_scale='Oranges', title=f"{target_country} 作品輸出地圖"), use_container_width=True)
                st.dataframe(export_stats, use_container_width=True)

        with tab5:
            top_titles = filtered_df.groupby(['show_title', 'Country']).size().reset_index(name='Weeks_On_Chart').sort_values('Weeks_On_Chart', ascending=False).head(10)
            fig = px.bar(top_titles, x='Weeks_On_Chart', y='show_title', orientation='h', color='Country', text_auto=True)
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with tab6:
            st.dataframe(filtered_df.groupby('Country')['show_title'].unique().apply(lambda x: ", ".join(x)).reset_index(name='Titles_List'), use_container_width=True)

        with tab7:
            st.dataframe(filtered_df, use_container_width=True)

        with st.expander("🤖 AI 市場總結"):
            if self.api_key and st.button("生成觀看國報告"):
                top_src = unique_counts.iloc[0]['Country'] if not unique_counts.empty else "無"
                prompt = f"分析 {target_country} 市場：最大來源{top_src}，請給出3點洞察。"
                st.markdown(ask_gemini(self.api_key, prompt, self.model_name))

    # -------------------------------------------------------------------------
    #  B. 製片國視角
    # -------------------------------------------------------------------------
    def analyze_producer(self, target_country):
        st.header(f"📦 文化輸出分析：{target_country} ({category_mode})")
        
        if target_country not in self.df['Country'].unique():
            st.warning(f"⚠️ 資料庫中沒有 {target_country} 製作的 {category_mode} 數據。")
            return

        filtered_df = self.df[self.df['Country'] == target_country].copy()

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "💎 輸出作品矩陣", 
            "🚀 最強傳播作品", 
            "🗺️ 全球版圖", 
            "🌍 海外市場表現",
            "📊 總週數排名", 
            "📑 詳細輸出清單", 
            "💾 原始數據"
        ])

        # --- [NEW] 1. 輸出作品矩陣 (最終版) ---
        with tab1:
            st.subheader(f"💎 {target_country} 輸出作品矩陣")
            st.markdown("""
            * **X軸**：海外上榜總週數 (續航力)
            * **Y軸**：**輸出國家數 (廣度)**
            * **大小**：**總觀看次數 (熱度)** (Log Scale)
            * **顏色**：海外最佳名次 (越紅越好)
            """)

            export_only_df = filtered_df[filtered_df['country_name'] != target_country].copy()
            
            if export_only_df.empty:
                st.info("該國作品僅在本國上榜，無海外輸出紀錄，無法繪製矩陣圖。")
            else:
                # 彙整 Views
                all_view_cols = [c for c in self.df.columns if 'Views' in c]
                all_view_cols.sort(reverse=True)
                
                unique_titles_view = self.df[['show_title'] + all_view_cols].drop_duplicates(subset=['show_title'])
                
                def get_latest_views(row):
                    for col in all_view_cols:
                        if pd.notna(row[col]) and row[col] > 0:
                            return row[col]
                    return 0

                unique_titles_view['Final_Views'] = unique_titles_view.apply(get_latest_views, axis=1)
                
                # 計算矩陣指標
                matrix_stats = export_only_df.groupby('show_title').agg(
                    Export_Countries=('country_name', 'nunique'),      # Y軸：輸出國家數
                    Weeks_Present_Overseas=('week', 'nunique'),        # X軸：海外上榜週數
                    Best_Rank_Overseas=('weekly_rank', 'min')          # 顏色：最佳名次
                ).reset_index()

                matrix_stats = pd.merge(matrix_stats, unique_titles_view[['show_title', 'Final_Views']], on='show_title', how='left')
                matrix_stats['Final_Views'] = matrix_stats['Final_Views'].fillna(0)
                
                # 計算 Log Views 供氣泡大小使用
                matrix_stats['Log_Views'] = np.log10(matrix_stats['Final_Views'] + 1)

                # 繪圖
                if not matrix_stats.empty:
                    fig_bubble = px.scatter(
                        matrix_stats,
                        x='Weeks_Present_Overseas', # X軸
                        y='Export_Countries',       # Y軸：改用國家數
                        size='Log_Views',           # 大小：改用觀看數(Log)
                        color='Best_Rank_Overseas', 
                        hover_name='show_title',
                        hover_data={'Log_Views': False, 'Final_Views': True}, # Tooltip 顯示真實數字
                        
                        range_color=[1, 10], 
                        color_continuous_scale='Reds_r',
                        size_max=60,
                        
                        title=f"{target_country} 作品輸出強弱分佈",
                        labels={
                            'Weeks_Present_Overseas': '海外上榜週數 (不重複)',
                            'Export_Countries': '輸出國家數',
                            'Final_Views': '總觀看次數',
                            'Best_Rank_Overseas': '最佳名次'
                        }
                    )
                    fig_bubble.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                    st.plotly_chart(fig_bubble, use_container_width=True)

                    st.markdown("##### 📌 矩陣數據詳表")
                    display_table = matrix_stats.sort_values('Final_Views', ascending=False)
                    display_table['Final_Views_Formatted'] = display_table['Final_Views'].apply(lambda x: "{:,.0f}".format(x))
                    st.dataframe(display_table[['show_title', 'Weeks_Present_Overseas', 'Export_Countries', 'Best_Rank_Overseas', 'Final_Views_Formatted']], use_container_width=True)
                else:
                    st.warning("數據計算後為空。")

        # --- 2. 最強傳播作品 ---
        with tab2:
            st.subheader("傳播力最強的作品")
            traveling = filtered_df.groupby('show_title')['country_name'].nunique().reset_index(name='Country_Count').sort_values('Country_Count', ascending=False).head(10)
            fig = px.bar(traveling, x='Country_Count', y='show_title', orientation='h', text_auto=True, title=f"輸出國家數最多的 Top 10 作品", color='Country_Count', color_continuous_scale='Oranges')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(traveling, use_container_width=True)

        # --- 3. 全球版圖 ---
        with tab3:
            st.subheader("全球輸出版圖")
            coverage = filtered_df.groupby('country_name')['show_title'].nunique().reset_index(name='Unique_Titles')
            st.plotly_chart(px.choropleth(coverage, locations="country_name", locationmode="country names", color="Unique_Titles", color_continuous_scale='Reds', title=f"{target_country} 作品覆蓋熱度圖"), use_container_width=True)
            st.dataframe(coverage.sort_values('Unique_Titles', ascending=False), use_container_width=True)

        # --- 4. 海外市場表現 ---
        with tab4:
            st.subheader("海外市場表現 (排除本國)")
            export_df = filtered_df[filtered_df['country_name'] != target_country]
            if export_df.empty: st.info("僅在本國上榜。")
            else:
                export_stats = export_df.groupby('country_name')['show_title'].nunique().reset_index(name='Exported_Titles').sort_values('Exported_Titles', ascending=False)
                c1, c2 = st.columns([2, 1])
                with c1: st.plotly_chart(px.choropleth(export_stats, locations="country_name", locationmode="country names", color="Exported_Titles", color_continuous_scale='Purples', title="海外輸出地圖"), use_container_width=True)
                with c2: st.dataframe(export_stats, use_container_width=True)

        # --- 5. 總週數排名 ---
        with tab5:
            st.subheader("各市場總熱度 (總週數)")
            raw_weeks = filtered_df['country_name'].value_counts().reset_index()
            raw_weeks.columns = ['Country', 'Total_Weeks']
            fig = px.bar(raw_weeks.head(20), x='Total_Weeks', y='Country', orientation='h', text_auto=True, title="上榜總週數 Top 20 市場")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(raw_weeks, use_container_width=True)

        # --- 6. 詳細輸出清單 ---
        with tab6:
            st.subheader("各市場上榜作品明細")
            st.dataframe(filtered_df.groupby('country_name')['show_title'].unique().apply(lambda x: ", ".join(x)).reset_index(name='Titles_List'), use_container_width=True)

        # --- 7. 原始數據 ---
        with tab7:
            st.dataframe(filtered_df, use_container_width=True)

        with st.expander("🤖 AI 輸出分析"):
            if self.api_key and st.button("生成製片國報告"):
                prompt = f"分析 {target_country} ({category_mode}) 文化輸出，請給3點洞察。"
                st.markdown(ask_gemini(self.api_key, prompt, self.model_name))

# ==========================================
# 6. 主程式執行邏輯
# ==========================================
analyzer = NetflixAnalyzerV6(df_main, gemini_api_key, selected_model)
analysis_mode = st.sidebar.radio("分析視角", ("觀看國 (Viewer)", "製片國 (Producer)"))

available_countries = sorted(list(set(df_main['country_name'].unique()) | set(df_main['Country'].unique())))
final_country_list = sorted([c for c in available_countries if c in TARGET_COUNTRIES])

if not final_country_list:
    st.warning("⚠️ 篩選後的資料中沒有包含您指定的目標國家。")
else:
    if "觀看國" in analysis_mode:
        selected_country = st.sidebar.selectbox("選擇觀看國家", final_country_list)
        if st.sidebar.button("開始分析"):
            analyzer.analyze_viewer(selected_country)
    else:
        selected_country = st.sidebar.selectbox("選擇製片國家", final_country_list)
        if st.sidebar.button("開始分析"):
            analyzer.analyze_producer(selected_country)
