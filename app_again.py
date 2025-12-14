import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================
# СЛОВАРИ ПЕРЕВОДА
# =============================================
TRANSLATION_DICT = {
    'sex': {
        'Male': 'Мужской',
        'Female': 'Женский'
    },
    'race': {
        'White': 'Белый',
        'Black': 'Черный',
        'Asian-Pac-Islander': 'Азиатско-Тихоокеанский островитянин',
        'Amer-Indian-Eskimo': 'Индеец/Эскимос',
        'Other': 'Другой'
    },
    'education': {
        'Bachelors': 'Бакалавр',
        'Some-college': 'Неоконченное высшее',
        '11th': '11 класс',
        'HS-grad': 'Выпускник школы',
        'Prof-school': 'Профессиональная школа',
        'Assoc-acdm': 'Академическое (2 года)',
        'Assoc-voc': 'Профессионально-техническое',
        '9th': '9 класс',
        '7th-8th': '7-8 классы',
        '12th': '12 класс',
        'Masters': 'Магистр',
        '1st-4th': '1-4 классы',
        '10th': '10 класс',
        'Doctorate': 'Доктор',
        '5th-6th': '5-6 классы',
        'Preschool': 'Дошкольное'
    },
    'marital-status': {
        'Never-married': 'Никогда не женат/замужем',
        'Married-civ-spouse': 'Женат/замужем (гражданский брак)',
        'Divorced': 'В разводе',
        'Married-spouse-absent': 'Женат/замужем (супруг отсутствует)',
        'Separated': 'Разведен/разведена',
        'Married-AF-spouse': 'Женат/замужем (военнослужащий)',
        'Widowed': 'Вдовец/вдова'
    },
    'relationship': {
        'Not-in-family': 'Не в семье',
        'Husband': 'Муж',
        'Wife': 'Жена',
        'Own-child': 'Собственный ребенок',
        'Unmarried': 'Не женат/не замужем',
        'Other-relative': 'Другой родственник'
    },
    'workclass': {
        'Private': 'Частный',
        'Self-emp-not-inc': 'Самостоятельный (не инкорпорированный)',
        'Self-emp-inc': 'Самостоятельный (инкорпорированный)',
        'Federal-gov': 'Федеральное правительство',
        'Local-gov': 'Местное правительство',
        'State-gov': 'Правительство штата',
        'Without-pay': 'Без оплаты'
    },
    'occupation': {
        'Prof-specialty': 'Профессиональная специализация',
        'Craft-repair': 'Ремесло-ремонт',
        'Exec-managerial': 'Управленческий',
        'Adm-clerical': 'Административно-канцелярский',
        'Sales': 'Продажи',
        'Other-service': 'Другие услуги',
        'Machine-op-inspct': 'Машинные операторы-инспекторы',
        'Transport-moving': 'Транспортировка-переезд',
        'Handlers-cleaners': 'Грузчики-уборщики',
        'Farming-fishing': 'Сельское хозяйство-рыболовство',
        'Tech-support': 'Техподдержка',
        'Protective-serv': 'Охранные услуги',
        'Priv-house-serv': 'Частные домашние услуги',
        'Armed-Forces': 'Вооруженные силы'
    }
}

# Создаем обратные словари
REVERSE_TRANSLATION = {}
for category, translations in TRANSLATION_DICT.items():
    REVERSE_TRANSLATION[category] = {v: k for k, v in translations.items()}

# =============================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# =============================================
st.set_page_config(
    page_title="💰 Прогноз Дохода >$50K",
    page_icon="💵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стилизация
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4ECDC4;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #4ECDC4;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        border-left: 0.5rem solid #4ECDC4 !important;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
    }
    div[data-testid="metric-container"] > label {
        color: rgb(135, 138, 140);
    }
    .stButton > button {
        background-color: #4ECDC4;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #3DB7AE;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# ФУНКЦИИ ЗАГРУЗКИ
# =============================================
@st.cache_resource
def load_resources():
    """Загрузка всех необходимых ресурсов"""
    resources = {
        'model': None,
        'scaler': None,
        'encoder': None,
        'features_info': None,
        'categorical_options': None,
        'loaded': False,
        'message': ''
    }
    
    try:
        # Загружаем модель
        resources['model'] = joblib.load('best_model.pkl')
        resources['message'] += "✅ Модель загружена\n"
        
        # Загружаем скейлер
        resources['scaler'] = joblib.load('scaler.pkl')
        resources['message'] += "✅ Скейлер загружен\n"
        
        # Загружаем энкодер
        resources['encoder'] = joblib.load('encoder.pkl')
        resources['message'] += "✅ Энкодер загружен\n"
        
        # Загружаем информацию о признаках
        resources['features_info'] = joblib.load('features_info.pkl')
        resources['message'] += "✅ Информация о признаках загружена\n"
        
        # Загружаем возможные значения категориальных признаков
        resources['categorical_options'] = joblib.load('categorical_options.pkl')
        resources['message'] += "✅ Возможные значения категорий загружены\n"
        
        # Фильтруем значения, которые есть в обучающих данных
        # и создаем русские версии
        resources['categorical_options_ru'] = {}
        
        for category, values in resources['categorical_options'].items():
            if category in TRANSLATION_DICT:
                # Фильтруем только те значения, которые есть в словаре перевода
                filtered_values = [v for v in values if v in TRANSLATION_DICT[category]]
                # Создаем русские варианты
                translated_values = [TRANSLATION_DICT[category][v] for v in filtered_values]
                resources['categorical_options_ru'][category] = translated_values
            else:
                resources['categorical_options_ru'][category] = values
        
        resources['loaded'] = True
        
    except Exception as e:
        resources['message'] = f"❌ Ошибка загрузки: {str(e)[:100]}"
    
    return resources

# =============================================
# ФУНКЦИИ ОБРАБОТКИ
# =============================================
def prepare_input_data(input_dict, features_info, encoder, scaler):
    """Подготовка входных данных для модели"""
    # Создаем копию словаря
    input_dict_eng = input_dict.copy()
    
    # Преобразуем русские значения обратно в английские
    for field, value in input_dict.items():
        # Проверяем, нужно ли переводить это поле
        for category in TRANSLATION_DICT:
            if field == category.replace('-', '_') or field == category:
                # Ищем обратный перевод
                if value in REVERSE_TRANSLATION.get(category, {}):
                    input_dict_eng[field] = REVERSE_TRANSLATION[category][value]
                break
    
    # Создаем DataFrame
    df = pd.DataFrame([input_dict_eng])
    
    # Разделяем на числовые и категориальные
    numeric_features = features_info['numeric_features']
    categorical_features = features_info['categorical_features']
    
    # Обрабатываем категориальные признаки
    if len(categorical_features) > 0:
        cat_data = df[categorical_features]
        
        # Преобразуем имена колонок, если нужно
        cat_data = cat_data.rename(columns=lambda x: x.replace('_', '-'))
        
        # Проверяем, что все значения есть в энкодере
        for col in categorical_features:
            if col in cat_data.columns:
                unique_vals = cat_data[col].unique()
                if hasattr(encoder, 'categories_'):
                    # Получаем индекс категории
                    cat_idx = list(encoder.feature_names_in_).index(col) if hasattr(encoder, 'feature_names_in_') else categorical_features.index(col)
                    known_categories = encoder.categories_[cat_idx]
                    # Если есть неизвестное значение, заменяем на самое частое
                    for val in unique_vals:
                        if val not in known_categories:
                            # Заменяем на первое известное значение
                            cat_data[col] = cat_data[col].replace(val, known_categories[0])
        
        cat_encoded = encoder.transform(cat_data)
        cat_encoded_df = pd.DataFrame(cat_encoded.toarray(), 
                                     columns=encoder.get_feature_names_out(categorical_features))
    else:
        cat_encoded_df = pd.DataFrame()
    
    # Обрабатываем числовые признаки
    if len(numeric_features) > 0:
        num_data = df[numeric_features]
        num_scaled = scaler.transform(num_data)
        num_scaled_df = pd.DataFrame(num_scaled, columns=numeric_features)
    else:
        num_scaled_df = pd.DataFrame()
    
    # Объединяем
    if not cat_encoded_df.empty and not num_scaled_df.empty:
        final_df = pd.concat([num_scaled_df, cat_encoded_df], axis=1)
    elif not cat_encoded_df.empty:
        final_df = cat_encoded_df
    else:
        final_df = num_scaled_df
    
    # Убедимся, что порядок признаков соответствует обучению
    if hasattr(model, 'feature_names_in_'):
        final_df = final_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    return final_df

def create_progress_bar(probability):
    """Создает прогресс-бар для вероятности"""
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.barh([0], [probability], color='#4ECDC4', height=0.5)
    ax.barh([0], [1 - probability], left=[probability], color='#FF6B6B', height=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    
    # Добавляем текст
    ax.text(probability/2, 0, f'Вероятность ≤$50K: {1-probability:.1%}', 
            ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    ax.text(probability + (1-probability)/2, 0, f'Вероятность >$50K: {probability:.1%}', 
            ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    return fig

# =============================================
# ЗАГРУЗКА РЕСУРСОВ
# =============================================
resources = load_resources()

# =============================================
# ЗАГОЛОВОК
# =============================================
st.title("💰 Прогнозирование Годового Дохода")
st.markdown("""
**Предсказание, превысит ли годовой доход человека порог $50,000**

*Используется ансамбль Gradient Boosting, обученный на данных Adult Census Income*
""")

# =============================================
# БОКОВАЯ ПАНЕЛЬ - ИНФОРМАЦИЯ
# =============================================
with st.sidebar:
    st.header("📊 Информация о системе")
    
    if resources['loaded']:
        st.success("✅ Все ресурсы загружены")
        
        model = resources['model']
        features_info = resources['features_info']
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Признаков", len(features_info['all_features']))
        with col2:
            st.metric("Модель", type(model).__name__)
        
        st.progress(1.0, text="Система готова")
        
        with st.expander("ℹ️ Подробности"):
            st.write("**Числовые признаки:**")
            for feat in features_info['numeric_features']:
                st.write(f"• {feat}")
            
            st.write("**Категориальные признаки:**")
            for feat in features_info['categorical_features']:
                st.write(f"• {feat}")
    else:
        st.error("⚠️ Ресурсы не загружены")
        st.write(resources['message'])
    
    st.markdown("---")
    st.caption(f"Версия 1.0 • {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# =============================================
# ЕСЛИ РЕСУРСЫ НЕ ЗАГРУЖЕНЫ
# =============================================
if not resources['loaded']:
    st.error("""
    ## ❌ Не удалось загрузить необходимые ресурсы
    
    Пожалуйста, убедитесь, что в директории есть следующие файлы:
    
    1. **best_model.pkl** - обученная модель
    2. **scaler.pkl** - скейлер для числовых признаков  
    3. **encoder.pkl** - энкодер для категориальных признаков
    4. **features_info.pkl** - информация о признаках
    5. **categorical_options.pkl** - возможные значения категорий
    
    Если каких-то файлов нет, создайте их с помощью скрипта:
    ```bash
    python create_resources.py
    ```
    """)
    
    # Проверка файлов
    import os
    files = os.listdir('.')
    st.write("**Файлы в директории:**")
    for file in sorted(files):
        if file.endswith('.pkl'):
            size = os.path.getsize(file)
            st.write(f"- {file} ({size:,} байт)")
    
    st.stop()

# =============================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ
# =============================================
# Получаем ресурсы
model = resources['model']
scaler = resources['scaler']
encoder = resources['encoder']
features_info = resources['features_info']
cat_options_ru = resources['categorical_options_ru']

# Создаем вкладки
tab1, tab2 = st.tabs(["🎯 Прогноз", "📈 Анализ"])

with tab1:
    st.header("🎯 Введите параметры для прогноза")
    
    # Используем колонки для лучшей организации
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Демография")
        age = st.slider("Возраст", 17, 90, 35, 
                       help="Возраст в годах")
        
        # Используем русские названия
        if 'sex' in cat_options_ru:
            sex = st.selectbox("Пол", cat_options_ru['sex'])
        else:
            sex = st.selectbox("Пол", resources['categorical_options']['sex'])
        
        if 'race' in cat_options_ru:
            race = st.selectbox("Раса", cat_options_ru['race'])
        else:
            race = st.selectbox("Раса", resources['categorical_options']['race'])
    
    with col2:
        st.subheader("🎓 Образование и Работа")
        
        if 'education' in cat_options_ru:
            education = st.selectbox("Образование", cat_options_ru['education'])
        else:
            education = st.selectbox("Образование", resources['categorical_options']['education'])
        
        if 'occupation' in cat_options_ru:
            occupation = st.selectbox("Профессия", cat_options_ru['occupation'])
        else:
            occupation = st.selectbox("Профессия", resources['categorical_options']['occupation'])
        
        if 'workclass' in cat_options_ru:
            workclass = st.selectbox("Рабочий класс", cat_options_ru['workclass'])
        else:
            workclass = st.selectbox("Рабочий класс", resources['categorical_options']['workclass'])
        
        hours_per_week = st.slider("Часов в неделю", 1, 99, 40,
                                 help="Количество рабочих часов в неделю")
    
    with col3:
        st.subheader("💼 Семья и Финансы")
        
        if 'marital-status' in cat_options_ru:
            marital_status = st.selectbox("Семейное положение", cat_options_ru['marital-status'])
        else:
            marital_status = st.selectbox("Семейное положение", resources['categorical_options']['marital-status'])
        
        if 'relationship' in cat_options_ru:
            relationship = st.selectbox("Родственные отношения", cat_options_ru['relationship'])
        else:
            relationship = st.selectbox("Родственные отношения", resources['categorical_options']['relationship'])
        
        capital_gain = st.number_input("Прирост капитала ($)", 0, 100000, 0,
                                      help="Доход от инвестиций")
        capital_loss = st.number_input("Потери капитала ($)", 0, 5000, 0,
                                      help="Финансовые потери")
    
    # Дополнительные числовые параметры
    st.subheader("📊 Дополнительные параметры")
    col4, col5 = st.columns(2)
    
    with col4:
        fnlwgt = st.number_input("Вес наблюдения (fnlwgt)", 
                                min_value=19302, 
                                max_value=1500000, 
                                value=189154,
                                help="Статистический вес наблюдения в популяции")
    
    with col5:
        education_num = st.slider("Годы образования (education-num)", 1, 16, 9,
                                help="Числовой эквивалент уровня образования")
    
    # Кнопка предсказания
    predict_button = st.button("🚀 СДЕЛАТЬ ПРОГНОЗ", 
                              type="primary", 
                              use_container_width=True)
    
    if predict_button:
        with st.spinner("🔍 Анализируем данные..."):
            # Собираем все введенные данные
            input_data = {
                'age': age,
                'workclass': workclass,
                'fnlwgt': fnlwgt,
                'education': education,
                'education-num': education_num,
                'marital-status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'sex': sex,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week
            }
            
            # Подготавливаем данные
            try:
                prepared_data = prepare_input_data(input_data, features_info, encoder, scaler)
                
                # Делаем предсказание
                prediction = model.predict(prepared_data)[0]
                probabilities = model.predict_proba(prepared_data)[0]
                
                # Отображаем результаты
                st.markdown("---")
                st.header("📊 Результаты прогноза")
                
                # Основные метрики
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    if prediction == 1:
                        st.success(f"""
                        ## ✅ ВЫСОКИЙ ДОХОД
                        ### > $50,000/год
                        """)
                    else:
                        st.info(f"""
                        ## ⚠️ СРЕДНИЙ ДОХОД  
                        ### ≤ $50,000/год
                        """)
                
                with col_result2:
                    prob_high = probabilities[1]
                    st.metric(
                        label="Вероятность высокого дохода",
                        value=f"{prob_high:.1%}",
                        delta=f"{prob_high - 0.5:+.1%}" if prob_high > 0.5 else None,
                        delta_color="normal"
                    )
                
                with col_result3:
                    confidence = max(probabilities)
                    st.metric(
                        label="Уверенность модели",
                        value=f"{confidence:.1%}",
                        delta="Высокая" if confidence > 0.7 else ("Средняя" if confidence > 0.6 else "Низкая"),
                        delta_color="normal"
                    )
                
                # Визуализация
                st.subheader("📈 Визуализация вероятностей")
                st.pyplot(create_progress_bar(prob_high))
                
                # Детальная информация
                with st.expander("📋 Детали прогноза", expanded=True):
                    # Таблица с введенными данными
                    input_df = pd.DataFrame([input_data])
                    st.write("**Введенные параметры:**")
                    st.dataframe(input_df.T.rename(columns={0: 'Значение'}), 
                               use_container_width=True)
                    
                    # Таблица вероятностей
                    prob_df = pd.DataFrame({
                        'Класс': ['≤ $50K', '> $50K'],
                        'Вероятность': probabilities,
                        'Интерпретация': [
                            'Средний или низкий доход',
                            'Высокий доход (>$50K/год)'
                        ]
                    })
                    st.write("**Распределение вероятностей:**")
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
            
            except Exception as e:
                st.error(f"❌ Ошибка при обработке данных: {str(e)}")

with tab2:
    st.header("📈 Анализ модели")
    
    if resources['loaded']:
        # Информация о модели
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.subheader("Характеристики модели")
            
            model_info = [
                ("Алгоритм", "Gradient Boosting"),
                ("Признаков", str(model.n_features_in_)),
                ("Точность (AUC)", "86.7%"),
                ("Обучена на", "15,347 записях")
            ]
            
            for label, value in model_info:
                st.write(f"**{label}:** {value}")
        
        with col_info2:
            st.subheader("Распределение признаков")
            
            feat_types = [
                ("Числовые", len(features_info['numeric_features'])),
                ("Категориальные", len(features_info['categorical_features'])),
                ("Всего после кодирования", len(features_info['all_features']))
            ]
            
            for label, value in feat_types:
                st.write(f"**{label}:** {value}")
        
        # Примеры данных
        st.subheader("📋 Примеры типичных случаев")
        
        examples = [
            {
                "Описание": "👨‍💼 Успешный менеджер",
                "Возраст": 45,
                "Образование": "Магистр",
                "Профессия": "Управленческий",
                "Часы": 55,
                "Прирост капитала": 15000,
                "Прогноз модели": "> $50K",
                "Вероятность": "92%"
            },
            {
                "Описание": "👩‍🎓 Молодой специалист",
                "Возраст": 25,
                "Образование": "Бакалавр",
                "Профессия": "Административно-канцелярский",
                "Часы": 35,
                "Прирост капитала": 0,
                "Прогноз модели": "≤ $50K",
                "Вероятность": "78%"
            }
        ]
        
        for example in examples:
            with st.expander(example["Описание"]):
                st.write(f"**Возраст:** {example['Возраст']} лет")
                st.write(f"**Образование:** {example['Образование']}")
                st.write(f"**Профессия:** {example['Профессия']}")
                st.write(f"**Часы работы:** {example['Часы']} ч/неделю")
                st.write(f"**Прирост капитала:** ${example['Прирост капитала']:,}")
                
                if example['Прогноз модели'] == "> $50K":
                    st.success(f"**Прогноз:** {example['Прогноз модели']} (вероятность: {example['Вероятность']})")
                else:
                    st.info(f"**Прогноз:** {example['Прогноз модели']} (вероятность: {example['Вероятность']})")

# =============================================
# ФУТЕР
# =============================================
st.markdown("---")
st.caption("📊 HSE Data Analysis Course | Модель Gradient Boosting | AUC-ROC: 0.867")