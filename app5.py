import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

# --- 1. Настройка страницы ---
st.set_page_config(
    page_title="Кластеризация научных статей: Спортивное питание",
    page_icon="🥦",
    layout="wide"
)

st.title("🥦 Кластеризация научных статей по спортивному питанию")
st.markdown("---")


# --- 2. Загрузка данных ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('LDA4_final.csv')
        # Приведение типов для стабильной работы
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
        if 'complexity' in df.columns:
            df['complexity'] = pd.to_numeric(df['complexity'], errors='coerce')
        if 'main_topic_probability' in df.columns:
            df['main_topic_probability'] = pd.to_numeric(df['main_topic_probability'], errors='coerce').fillna(0)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")
        return pd.DataFrame()


df = load_data()

if df.empty:
    st.stop()

# --- 3. Боковая панель (Фильтры) ---
with st.sidebar:
    st.header(" Фильтры")

    all_authors = set()
    if 'authors_short' in df.columns:
        raw_authors = df['authors_short'].dropna().astype(str).tolist()
        for row in raw_authors:
            names = [name.strip() for name in row.split(',')]
            all_authors.update(names)

    sorted_authors = sorted(list(all_authors))

    st.write("**Темы публикаций:**")
    available_topics = sorted(df['main_topic_name'].dropna().unique())

    min_y = int(df['year'].min()) if not df['year'].isna().all() else 2000
    max_y = int(df['year'].max()) if not df['year'].isna().all() else 2024

    c_min = int(df['complexity'].min()) if not df['complexity'].isna().all() else 0
    c_max = int(df['complexity'].max()) if not df['complexity'].isna().all() else 5000

    def reset_filters():
        st.session_state['authors_filter'] = []
        st.session_state['year_range'] = (min_y, max_y)
        st.session_state['complexity_range'] = (c_min, c_max)
        for t in available_topics:
            st.session_state[f"filter_{t}"] = True
        st.session_state['curr_page'] = 1

    selected_authors = st.multiselect(
        "Авторы",
        options=sorted_authors[:1000],
        default=[],
        key="authors_filter"
    )

    selected_topics = []
    for t in available_topics:
        if st.checkbox(t, value=True, key=f"filter_{t}"):
            selected_topics.append(t)

    year_range = st.slider("Год публикации", min_y, max_y, (min_y, max_y), key="year_range")

    complexity_range = st.slider("Объем (слов)", c_min, c_max, (c_min, c_max), key="complexity_range")

    st.button("Сбросить все фильтры", on_click=reset_filters)

# --- 4. Применение фильтрации ---
filtered_df = df.copy()

if selected_authors:
    mask = filtered_df['authors_short'].apply(lambda x: any(auth in str(x) for auth in selected_authors))
    filtered_df = filtered_df[mask]

if selected_topics:
    filtered_df = filtered_df[filtered_df['main_topic_name'].isin(selected_topics)]

filtered_df = filtered_df[
    (filtered_df['year'] >= year_range[0]) &
    (filtered_df['year'] <= year_range[1]) &
    (filtered_df['complexity'] >= complexity_range[0]) &
    (filtered_df['complexity'] <= complexity_range[1])
    ]

# --- 5. Визуализация ---
st.header("⩫ Визуализация кластеров статей")

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric(" Всего статей", len(filtered_df))
col_m2.metric(" Тем в выборке", len(filtered_df['main_topic_name'].unique()))
col_m3.metric(" Годы", f"{year_range[0]}-{year_range[1]}")
col_m4.metric(" Средняя сложность", f"{int(filtered_df['complexity'].mean()) if not filtered_df.empty else 0} слов")

st.subheader("⩫ Интерактивная карта статей")
colors = px.colors.qualitative.Set2


topic_cols = [f'topic_{i+1}_prob' for i in range(5)]
X = df[topic_cols].values


# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
coords_tsne = tsne.fit_transform(X)


df['tsne_x'] = coords_tsne[:, 0]
df['tsne_y'] = coords_tsne[:, 1]


df['authors_short_display'] = df['authors_short'].fillna('Неизвестно')
df['year_display'] = df['year'].fillna(0).astype(int)
df['complexity_display'] = df['complexity'].fillna(0).astype(int)

fig_tsne = px.scatter(
    filtered_df,
    x='tsne_x',
    y='tsne_y',
    color='main_topic_name',
    hover_name='title',
    hover_data={
        'authors_short_display': True,
        'main_topic_name': True,
        'main_topic_probability': ':.2f',
        'year_display': True,
        'complexity_display': True,
        'tsne_x': False,
        'tsne_y': False
    },
    color_discrete_sequence=px.colors.qualitative.Set2,
    title="t-SNE визуализация LDA",
    opacity=0.7,
    #height=700
)



fig_tsne.update_traces(
        marker=dict(size=8, line=dict(width=0.8, color='white')),
        hovertemplate="<b>Название статьи:</b> %{hovertext}</b><br><br>" +
                      "<b>Авторы:</b> %{customdata[0]}<br>" +
                      "<b>Тема:</b> %{customdata[1]}<br>" +
                      "<b>Уверенность модели:</b> %{customdata[2]:.1%}<br>" +
                      "<b>Год:</b> %{customdata[3]}<br>" +
                      "<b>Объем:</b> %{customdata[4]} слов" +
                      "<extra></extra>"
)

fig_tsne.update_layout(
    xaxis_title=f't-SNE 1',
    yaxis_title=f't-SNE 2',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=1.05,
        font=dict(size=14),
        title=dict(text="<b>Темы LDA</b>")
    ),
    margin=dict(r=150, l=80, b=80, t=80),
    hoverlabel=dict(
        bgcolor="white",
        font_size=14,
        font_family="Arial",
        font_color="black",
        bordercolor="gray"
    )
)

st.plotly_chart(fig_tsne, use_container_width=True)

st.header("⩫ Статистика по темам")
col_left, col_right = st.columns([1, 0.68])

filtered_df['year'] = filtered_df['year'].astype(int)

if not filtered_df.empty:
    with col_left:
        topic_counts = filtered_df['main_topic_name'].value_counts().reset_index()
        topic_counts.columns = ['Тема', 'Количество']
        fig_bar = px.bar(
            topic_counts, x='Количество', y='Тема', orientation='h',
            title="Распределение статей по темам", color='Тема',
            color_discrete_sequence=colors, text='Количество'
        )
        fig_bar.update_traces(
            hovertemplate=" <b>Статей в теме:</b> %{x}<br> <b>Доля:</b> %{customdata:.1%}<extra></extra>",
            customdata=topic_counts['Количество'] / topic_counts['Количество'].sum(),
            textposition='outside'
        )
        fig_bar.update_layout(showlegend=False, height=400, xaxis_title="Количество статей", yaxis_title="Темы")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        yearly_topics = filtered_df.groupby(['year', 'main_topic_name']).size().reset_index(name='count')
        fig_area = px.area(
            yearly_topics, x='year', y='count', color='main_topic_name',
            title="Динамика публикаций по годам", color_discrete_sequence=colors
        )
        fig_area.update_traces(
            hovertemplate="<b style='font-size: 12px;'> %{fullData.name}</b><br><br><b>Год:</b> %{x}<br><b>Статей:</b> %{y}<extra></extra>"
        )
        fig_area.update_layout(height=400, xaxis_title="Год", yaxis_title="Количество статей",
                               legend_title="<b>Темы</b>")
        st.plotly_chart(fig_area, use_container_width=True)

# --- 6. Детальный анализ тем (Восстановлено из app2.py) ---
st.header("⩫ Детальный анализ тем")

available_themes = sorted(filtered_df['main_topic_name'].unique())

if available_themes:
    selected_topic_detailed = st.selectbox(
        "Выберите тему для детального просмотра:",
        options=available_themes,
        key="detailed_analysis_select"
    )

    if selected_topic_detailed:
        topic_df = filtered_df[filtered_df['main_topic_name'] == selected_topic_detailed]

        st.subheader(f"⋇ {selected_topic_detailed}")
        if 'topic_interpretation' in topic_df.columns and not topic_df.empty:
            st.caption(str(topic_df.iloc[0]['topic_interpretation']))

        col_meta1, col_meta3 = st.columns(2)
        with col_meta1:
            st.metric("Количество статей", len(topic_df))
        with col_meta3:
            t_min_y = int(topic_df['year'].min()) if not topic_df['year'].isna().all() else "Н/Д"
            t_max_y = int(topic_df['year'].max()) if not topic_df['year'].isna().all() else "Н/Д"
            st.metric("Диапазон лет", f"{t_min_y}-{t_max_y}")

        st.subheader("⩫ Статьи по теме")

        display_cols = ['title', 'authors_short', 'year', 'journal', 'main_topic_probability']
        existing_cols = [c for c in display_cols if c in topic_df.columns]
        display_df = topic_df[existing_cols].copy()

        rename_dict = {
            'title': 'Название',
            'authors_short': 'Авторы',
            'year': 'Год',
            'journal': 'Журнал',
            'main_topic_probability': 'Вероятность'
        }
        display_df.rename(columns=rename_dict, inplace=True)
        display_df = display_df.sort_values('Вероятность', ascending=False)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config={
                "Название": st.column_config.TextColumn(width="large"),
                "Авторы": st.column_config.TextColumn(width="medium"),
                "Вероятность": st.column_config.NumberColumn(format="%.2f")
            }
        )

        csv = topic_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Скачать статьи по теме {selected_topic_detailed} (CSV)",
            data=csv,
            file_name=f'{selected_topic_detailed.replace(" ", "_")}.csv',
            mime='text/csv',
            use_container_width=True
        )
else:
    st.info("Нет доступных тем для анализа. Проверьте фильтры в боковой панели.")

# --- 7. Поиск и Рекомендации ---
st.header("⩫ Поиск и Рекомендации")
tab1, tab2 = st.tabs([" Поиск", " Похожие статьи"])

with tab1:
    s_col1, s_col2, s_col3 = st.columns([2, 1, 1])
    search_q = s_col1.text_input("Поиск по тексту:", placeholder="Ключевые слова...", key="search_input")
    sort_by = s_col2.selectbox("Сортировка:", ["Новые", "Старые", "А-Я", "Вероятность темы"])
    page_size = s_col3.select_slider("На странице:", options=[5, 10, 20], value=10)

    res = filtered_df.copy()
    if search_q:
        res = res[
            res['clean_abstract'].str.contains(search_q, case=False, na=False) | res['title'].str.contains(search_q,
                                                                                                           case=False,
                                                                                                           na=False)]

    sort_map = {"Новые": ("year", False), "Старые": ("year", True), "А-Я": ("title", True),
                "Вероятность темы": ("main_topic_probability", False)}
    res = res.sort_values(by=sort_map[sort_by][0], ascending=sort_map[sort_by][1])

    total = len(res)
    pages = (total // page_size) + (1 if total % page_size > 0 else 0)
    if 'curr_page' not in st.session_state: st.session_state.curr_page = 1

    if total > 0:
        start = (st.session_state.curr_page - 1) * page_size
        for _, row in res.iloc[start: start + page_size].iterrows():
            y_val = int(row['year']) if pd.notnull(row['year']) else '?'
            with st.expander(f"{row['title']} ({y_val})"):
                st.write(f"**Авторы:** {row['authors_short']}")
                st.write(f"**Журнал:** {row.get('journal', 'N/A')}")
                st.write(f"**Тема:** {row['main_topic_name']} (p={row['main_topic_probability']:.2f})")
                st.markdown("**Аннотация (полный текст):**")
                st.write(str(row['clean_abstract']))

        st.markdown("<br>", unsafe_allow_html=True)
        nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 0.5, 2, 0.5, 1])
        with nav_col1:
            if st.button("⬅ Назад", disabled=st.session_state.curr_page <= 1, use_container_width=True):
                st.session_state.curr_page -= 1
                st.rerun()
        with nav_col3:
            st.markdown(
                f"<p style='text-align:center; font-weight: bold;'>Страница {st.session_state.curr_page} из {max(pages, 1)}</p>",
                unsafe_allow_html=True)
        with nav_col5:
            if st.button("Вперед ➡", disabled=st.session_state.curr_page >= pages, use_container_width=True):
                st.session_state.curr_page += 1
                st.rerun()
    else:
        st.info("Статьи не найдены.")

with tab2:
    st.subheader("Рекомендации на основе контента")
    if not filtered_df.empty:
        def fmt(r):
            y = int(r['year']) if pd.notnull(r['year']) else "Н/Д"
            return f"{str(r['title'])[:70]}... ({y}) | ID:{r.name}"

        titles_options = filtered_df.apply(fmt, axis=1).tolist()
        pick = st.selectbox("Выберите статью-образец:", [""] + titles_options, key="rec_box")

        if pick:
            idx = int(pick.split("ID:")[1])
            target = df.loc[idx]

            coord_cols = ['tsne_x', 'tsne_y']
            cands = filtered_df[filtered_df.index != idx].copy()
            cands = cands.dropna(subset=coord_cols)

            if pd.notnull(target['tsne_x']) and pd.notnull(target['tsne_y']) and not cands.empty:
                cands['dist'] = (
                    (cands['tsne_x'] - target['tsne_x']) ** 2 +
                    (cands['tsne_y'] - target['tsne_y']) ** 2
                ) ** 0.5

                dist_scale = float(cands['dist'].quantile(0.95)) if not cands['dist'].empty else 0.0
                if dist_scale <= 0:
                    dist_scale = float(cands['dist'].max()) if not cands['dist'].empty else 1.0

                recs = cands.sort_values('dist').head(5)

                st.write(f"### Похожие на: *{target['title']}*")
                for _, r in recs.iterrows():
                    score = max(0.0, 100.0 * (1.0 - (r['dist'] / dist_scale)))
                    with st.expander(
                            f" Сходство {score:.1f}% — {r['title']} ({int(r['year']) if pd.notnull(r['year']) else '?'})"):
                        st.markdown(
                            f"**Почему рекомендовано:** Статья находится в непосредственной близости в кластере *{r['main_topic_name']}*.")
                        st.write(f"**Авторы:** {r['authors_short']}")
                        st.markdown("**Полная аннотация:**")
                        st.write(str(r['clean_abstract']))
            else:
                st.info("Недостаточно данных для расчета похожих статей по координатам.")
    else:
        st.warning("Нет данных для рекомендаций.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Кластеризация научных статей по спортивному питанию<br>
    </div>
    """,
    unsafe_allow_html=True
)
