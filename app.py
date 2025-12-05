import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

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

st.set_page_config(page_title="Netflix 數據戰情室 V4.1", layout="wide")
st.title("🎬 Netflix 深度數據分析系統 (含雙向視角)")

# ==========================================
# 2. 資料讀取 (含 Films/TV 切換邏輯)
# ==========================================
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df['week'] = pd.to_datetime(df['week'])
        df['Week_Str'] = df['week'].dt.strftime('%Y-%m-%d')
        return df
    except FileNotFoundError:
        st.error(f"找不到檔案 '{file_path}'")
        return pd.DataFrame()

# 讀取完整檔案
df_raw = load_data('總表(new)_20251027.csv')

if df_raw.empty:
    st.stop()

# ==========================================
# 3. 側邊欄設定
# ==========================================
st.sidebar.header("⚙️ 參數設定")

# 類別切換
category_mode = st.sidebar.radio("內容類別", ("Films", "TV"), index=0)

# 資料篩選
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
class NetflixAnalyzerV4:
    def __init__(self, df, api_key, model_name):
        self.df = df.copy()
        self.api_key = api_key
        self.model_name = model_name

    # -------------------------------------------------------------------------
    #  A. 觀看國視角 (Viewer Perspective)
    # -------------------------------------------------------------------------
    def analyze_viewer(self, target_country):
        st.header(f"🌍 消費市場分析：{target_country} ({category_mode})")
        
        # 檢查該國是否為觀看國
        if target_country not in self.df['country_name'].unique():
            st.warning(f"⚠️ 資料庫中沒有 {target_country} 的觀看數據。")
            return

        filtered_df = self.df[self.df['country_name'] == target_country].copy()
        domestic_export_df = self.df[self.df['Country'] == target_country].copy()

        # 分頁規劃
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 來源排名(量)", 
            "🏆 冠軍來源國", 
            "🗺️ 來源地圖", 
            "🚀 本國輸出表現",
            "🔥 熱門作品", 
            "📑 詳細清單", 
            "💾 原始數據"
        ])

        # --- 1. 整體作品數排名 (Total Unique Bar) ---
        with tab1:
            st.subheader("各國輸入作品量排名")
            
            unique_counts = filtered_df.groupby('Country')['show_title'].nunique().reset_index(name='Unique_Titles')
            unique_counts = unique_counts.sort_values('Unique_Titles', ascending=False)
            
            fig = px.bar(
                unique_counts, 
                x='Unique_Titles', y='Country', orientation='h',
                text_auto=True, 
                title=f"{target_country} 的內容供應國排名 (依片量)",
                color='Unique_Titles', color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### 📌 詳細數據表")
            st.dataframe(unique_counts, use_container_width=True)

        # --- 2. [修正] 冠軍來源國 (Source of #1) ---
        with tab2:
            st.subheader("誰統治了冠軍寶座？")
            st.caption("統計在此市場獲得「第 1 名」次數最多的製片國家。")
            
            # 篩選第1名
            rank1_df = filtered_df[filtered_df['weekly_rank'] == 1]
            if rank1_df.empty:
                st.info("無冠軍數據")
            else:
                rank1_counts = rank1_df['Country'].value_counts().reset_index()
                rank1_counts.columns = ['Producer_Country', 'Weeks_at_No1']
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    fig_pie = px.pie(
                        rank1_counts, 
                        values='Weeks_at_No1', 
                        names='Producer_Country',
                        title='冠軍週數佔比'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2:
                    st.dataframe(rank1_counts, use_container_width=True)
                    
                st.markdown("##### 📌 冠軍作品明細")
                # [FIX] 這裡修正為 groupby('Country')，因為 rank1_df 裡面還是用 Country 這個名字
                rank1_titles = rank1_df.groupby('Country')['show_title'].unique().apply(lambda x: ", ".join(x)).reset_index(name='Champion_Titles')
                st.dataframe(rank1_titles, use_container_width=True)

        # --- 3. 上榜作品數地圖 ---
        with tab3:
            st.subheader("內容來源全球地圖")
            
            # 如果 tab1 沒跑可能導致 unique_counts 不存在，這裡重新算一次確保安全
            unique_counts = filtered_df.groupby('Country')['show_title'].nunique().reset_index(name='Unique_Titles')
            
            fig_map = px.choropleth(
                unique_counts,
                locations="Country", locationmode="country names",
                color="Unique_Titles",
                color_continuous_scale='Greens',
                title=f"{target_country} 的內容進口地圖"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            st.markdown("##### 📌 詳細數據表")
            st.dataframe(unique_counts, use_container_width=True)

        # --- 4. 本國輸出表現 (Domestic Export) ---
        with tab4:
            st.subheader(f"{target_country} 自製內容流向何方？")
            st.caption(f"在分析 {target_country} 市場的同時，看看該國製作的內容賣到了哪些國家。")
            
            if domestic_export_df.empty:
                st.warning(f"{target_country} 在資料集中沒有製作任何 {category_mode} 內容。")
            else:
                export_stats = domestic_export_df.groupby('country_name')['show_title'].nunique().reset_index(name='Titles_Count')
                export_stats = export_stats.sort_values('Titles_Count', ascending=False)
                
                fig_exp_map = px.choropleth(
                    export_stats,
                    locations="country_name", locationmode="country names",
                    color="Titles_Count",
                    color_continuous_scale='Oranges',
                    title=f"{target_country} 作品輸出地圖"
                )
                st.plotly_chart(fig_exp_map, use_container_width=True)
                
                st.markdown("##### 📌 輸出國家數據表")
                st.dataframe(export_stats, use_container_width=True)
                
                st.markdown("##### 📌 輸出作品清單 (各國看了什麼)")
                export_detail = domestic_export_df.groupby('country_name')['show_title'].unique().apply(lambda x: ", ".join(x)).reset_index(name='Exported_Titles')
                st.dataframe(export_detail, use_container_width=True)

        # --- 5. 最受歡迎作品 ---
        with tab5:
            st.subheader("霸榜最久的作品 Top 10")
            
            top_titles = filtered_df.groupby(['show_title', 'Country']).size().reset_index(name='Weeks_On_Chart')
            top_titles = top_titles.sort_values('Weeks_On_Chart', ascending=False).head(10)
            
            fig = px.bar(
                top_titles, x='Weeks_On_Chart', y='show_title', orientation='h',
                color='Country', text_auto=True
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### 📌 詳細數據表")
            st.dataframe(top_titles, use_container_width=True)

        # --- 6. 詳細清單 ---
        with tab6:
            st.subheader("各來源國作品明細")
            detail_list = filtered_df.groupby('Country')['show_title'].unique().apply(lambda x: ", ".join(x)).reset_index(name='Titles_List')
            st.dataframe(detail_list, use_container_width=True)

        # --- 7. 原始數據 ---
        with tab7:
            st.dataframe(filtered_df, use_container_width=True)

        # AI Insight
        with st.expander("🤖 AI 市場總結"):
            if self.api_key and st.button("生成觀看國報告"):
                unique_counts = filtered_df.groupby('Country')['show_title'].nunique().reset_index(name='Unique_Titles').sort_values('Unique_Titles', ascending=False)
                top_source = unique_counts.iloc[0]['Country'] if not unique_counts.empty else "無"
                
                # 重新檢查 rank1
                rank1_df = filtered_df[filtered_df['weekly_rank'] == 1]
                champion_source = "無"
                if not rank1_df.empty:
                    champion_source = rank1_df['Country'].value_counts().idxmax()
                
                prompt = f"分析 {target_country} ({category_mode}) 市場：\n最大內容來源：{top_source}\n冠軍週數最多國：{champion_source}\n請給出3點洞察。"
                st.markdown(ask_gemini(self.api_key, prompt, self.model_name))

    # -------------------------------------------------------------------------
    #  B. 製片國視角 (Producer Perspective)
    # -------------------------------------------------------------------------
    def analyze_producer(self, target_country):
        st.header(f"📦 文化輸出分析：{target_country} ({category_mode})")
        
        if target_country not in self.df['Country'].unique():
            st.warning(f"⚠️ 資料庫中沒有 {target_country} 製作的 {category_mode} 數據。")
            return

        filtered_df = self.df[self.df['Country'] == target_country].copy()

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🚀 最強傳播作品", 
            "🗺️ 全球版圖", 
            "🌍 海外市場表現",
            "📊 總週數排名", 
            "📑 詳細輸出清單", 
            "💾 原始數據"
        ])

        # --- 1. 最強傳播作品 ---
        with tab1:
            st.subheader("傳播力最強的作品")
            
            traveling = filtered_df.groupby('show_title')['country_name'].nunique().reset_index(name='Country_Count')
            traveling = traveling.sort_values('Country_Count', ascending=False).head(10)
            
            fig = px.bar(
                traveling, x='Country_Count', y='show_title', orientation='h',
                text_auto=True, title=f"輸出國家數最多的 Top 10 作品",
                color='Country_Count', color_continuous_scale='Oranges'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### 📌 詳細數據表")
            st.dataframe(traveling, use_container_width=True)

        # --- 2. 全球版圖 ---
        with tab2:
            st.subheader("全球輸出版圖")
            
            coverage = filtered_df.groupby('country_name')['show_title'].nunique().reset_index(name='Unique_Titles')
            
            fig_map = px.choropleth(
                coverage,
                locations="country_name", locationmode="country names",
                color="Unique_Titles",
                color_continuous_scale='Reds',
                title=f"{target_country} 作品覆蓋熱度圖"
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
            st.markdown("##### 📌 詳細數據表")
            st.dataframe(coverage.sort_values('Unique_Titles', ascending=False), use_container_width=True)

        # --- 3. 海外市場表現 ---
        with tab3:
            st.subheader("海外市場表現 (排除本國)")
            
            export_df = filtered_df[filtered_df['country_name'] != target_country]
            
            if export_df.empty:
                st.info("僅在本國上榜。")
            else:
                export_stats = export_df.groupby('country_name')['show_title'].nunique().reset_index(name='Exported_Titles')
                export_stats = export_stats.sort_values('Exported_Titles', ascending=False)
                
                c1, c2 = st.columns([2, 1])
                with c1:
                    fig_export = px.choropleth(
                        export_stats,
                        locations="country_name", locationmode="country names",
                        color="Exported_Titles",
                        color_continuous_scale='Purples',
                        title="海外輸出地圖"
                    )
                    st.plotly_chart(fig_export, use_container_width=True)
                with c2:
                    st.dataframe(export_stats, use_container_width=True)

        # --- 4. 總週數排名 ---
        with tab4:
            st.subheader("各市場總熱度 (總週數)")
            
            raw_weeks = filtered_df['country_name'].value_counts().reset_index()
            raw_weeks.columns = ['Country', 'Total_Weeks']
            
            fig = px.bar(
                raw_weeks.head(20), x='Total_Weeks', y='Country', orientation='h',
                text_auto=True, title="上榜總週數 Top 20 市場"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("##### 📌 詳細數據表")
            st.dataframe(raw_weeks, use_container_width=True)

        # --- 5. 詳細輸出清單 ---
        with tab5:
            st.subheader("各市場上榜作品明細")
            detail_list = filtered_df.groupby('country_name')['show_title'].unique().apply(lambda x: ", ".join(x)).reset_index(name='Titles_List')
            st.dataframe(detail_list, use_container_width=True)

        # --- 6. 原始數據 ---
        with tab6:
            st.dataframe(filtered_df, use_container_width=True)

        # AI Insight
        with st.expander("🤖 AI 輸出分析"):
            if self.api_key and st.button("生成製片國報告"):
                traveling = filtered_df.groupby('show_title')['country_name'].nunique().reset_index(name='Country_Count').sort_values('Country_Count', ascending=False)
                coverage = filtered_df.groupby('country_name')['show_title'].nunique().reset_index(name='Unique_Titles').sort_values('Unique_Titles', ascending=False)
                
                top_work = traveling.iloc[0]['show_title'] if not traveling.empty else "無"
                top_market = coverage.iloc[0]['country_name'] if not coverage.empty else "無"
                
                prompt = f"分析 {target_country} ({category_mode}) 的文化輸出：\n傳播最廣作品：{top_work}\n最大出口市場：{top_market}\n請給出3點輸出策略洞察。"
                st.markdown(ask_gemini(self.api_key, prompt, self.model_name))

# ==========================================
# 6. 主程式執行邏輯
# ==========================================
analyzer = NetflixAnalyzerV4(df_main, gemini_api_key, selected_model)

analysis_mode = st.sidebar.radio("分析視角", ("觀看國 (Viewer)", "製片國 (Producer)"))

# 篩選國家
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