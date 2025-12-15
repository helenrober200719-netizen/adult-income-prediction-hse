import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_progress_bar(probability):
    """Создает красивый график-прогресс бар"""
    fig, ax = plt.subplots(figsize=(10, 1.5))
    
    # Создаем горизонтальный бар
    bars = ax.barh(['Вероятность'], [probability], color='#4ECDC4', height=0.4)
    ax.barh(['Вероятность'], [1-probability], left=[probability], 
            color='#f0f2f6', height=0.4)
    
    # Настройки графика
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    
    # Добавляем текст в середину бара
    ax.text(probability/2, 0, f'{probability:.1%}', 
            ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    # Добавляем пороговую линию на 50%
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(0.5, 0.5, ' Порог 50%', transform=ax.get_xaxis_transform(), 
            color='red', va='center', fontsize=10)
    
    # Убираем рамки
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    
    plt.tight_layout()
    return fig

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
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        border-left: 0.5rem solid #4ECDC4 !important;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
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
# ФУНКЦИИ ДЕМО-РЕЖИМА
# =============================================
@st.cache_resource
def load_demo_resources():
    """Создание демо-ресурсов в памяти"""
    import numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
    resources = {
        'model': None,
        'scaler': None,
        'encoder': None,
        'features_info': None,
        'categorical_options': None,
        'categorical_options_ru': None,
        'loaded': False,
        'message': '',
        'demo_mode': True
    }
    
    try:
        # Создаем демо-модель
        class DemoModel:
            def __init__(self):
                self.random_seed = 42
            
            def predict(self, X):
                np.random.seed(self.random_seed)
                # Простая логика: возраст > 40 и образование > 12 -> высокий доход
                predictions = []
                for sample in X:
                    if len(sample) >= 5:
                        age = sample[0]
                        education = sample[4]
                        if age > 40 and education > 12:
                            predictions.append(1)
                        else:
                            predictions.append(0)
                    else:
                        predictions.append(0)
                return np.array(predictions)
            
            def predict_proba(self, X):
                np.random.seed(self.random_seed)
                prob_high = []
                for sample in X:
                    base_prob = 0.3
                    if len(sample) >= 5:
                        age = sample[0]
                        education = sample[4]
                        if age > 40:
                            base_prob += 0.3
                        if education > 12:
                            base_prob += 0.3
                    base_prob = min(base_prob, 0.95)
                    base_prob = max(base_prob, 0.05)
                    prob_high.append(base_prob)
                
                prob_high = np.array(prob_high).reshape(-1, 1)
                return np.hstack([1 - prob_high, prob_high])
        
        # Создаем и "обучаем" скейлер на синтетических данных
        scaler = StandardScaler()
        dummy_numeric = np.array([
            [30, 200000, 9, 0, 0, 40],
            [50, 300000, 13, 10000, 0, 50],
            [25, 150000, 10, 0, 1000, 35]
        ])
        scaler.fit(dummy_numeric)
        
        # Создаем и "обучаем" энкодер
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        dummy_categorical = pd.DataFrame({
            'workclass': ['Private', 'Federal-gov', 'Self-emp-not-inc'],
            'education': ['Bachelors', 'Masters', 'HS-grad'],
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced'],
            'occupation': ['Exec-managerial', 'Prof-specialty', 'Adm-clerical'],
            'relationship': ['Not-in-family', 'Husband', 'Own-child'],
            'race': ['White', 'Black', 'Asian-Pac-Islander'],
            'sex': ['Male', 'Female', 'Male']
        })
        encoder.fit(dummy_categorical)
        
        # Метаданные о признаках
        resources['features_info'] = {
            'numeric_features': ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
            'categorical_features': ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex'],
            'all_features': []
        }
        
        # Категориальные значения (английские)
        resources['categorical_options'] = {
            'sex': ['Male', 'Female'],
            'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
            'education': ['Bachelors', 'Some-college', 'HS-grad', 'Masters', 'Assoc-voc', 'Assoc-acdm'],
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated'],
            'relationship': ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried'],
            'workclass': ['Private', 'Self-emp-not-inc', 'Federal-gov', 'Local-gov', 'State-gov'],
            'occupation': ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales']
        }
        
        # Создаем русские версии
        resources['categorical_options_ru'] = {}
        for category, eng_values in resources['categorical_options'].items():
            if category in TRANSLATION_DICT:
                rus_values = []
                for eng_val in eng_values:
                    if eng_val in TRANSLATION_DICT[category]:
                        rus_values.append(TRANSLATION_DICT[category][eng_val])
                    else:
                        rus_values.append(eng_val)
                resources['categorical_options_ru'][category] = rus_values
            else:
                resources['categorical_options_ru'][category] = eng_values
        
        # Присваиваем объекты
        resources['model'] = DemoModel()
        resources['scaler'] = scaler
        resources['encoder'] = encoder
        resources['loaded'] = True
        resources['message'] = "✅ Демо-ресурсы успешно созданы\n"
        
    except Exception as e:
        resources['message'] = f"❌ Ошибка создания демо-ресурсов: {str(e)}"
    
    return resources

# =============================================
# ФУНКЦИЯ ПОДГОТОВКИ ДАННЫХ
# =============================================
def prepare_demo_input(input_dict, features_info, encoder, scaler):
    """Безопасная подготовка данных для демо-режима"""
    import pandas as pd
    import numpy as np
    
    # =============================================
    # 1. БЕЗОПАСНОЕ ПРЕОБРАЗОВАНИЕ ТИПОВ
    # =============================================
    
    # Копируем словарь, чтобы не менять оригинал
    processed_dict = input_dict.copy()
    
    # Определяем, какие поля должны быть числами
    numeric_fields = ['age', 'fnlwgt', 'education-num', 
                     'capital-gain', 'capital-loss', 'hours-per-week']
    
    # Значения по умолчанию для числовых полей
    defaults = {
        'age': 35,
        'fnlwgt': 189154,
        'education-num': 9,
        'capital-gain': 0,
        'capital-loss': 0,
        'hours-per-week': 40
    }
    
    # Преобразуем все числовые поля
    for field in numeric_fields:
        if field in processed_dict:
            value = processed_dict[field]
            
            # Если значение - строка, преобразуем в число
            if isinstance(value, str):
                try:
                    # Очищаем строку от пробелов и запятых
                    clean_val = str(value).replace(',', '').replace(' ', '').strip()
                    if clean_val == '':
                        processed_dict[field] = defaults[field]
                    else:
                        # Пробуем преобразовать в число
                        num_val = float(clean_val)
                        # Для некоторых полей лучше целые числа
                        if field in ['age', 'education-num', 'hours-per-week']:
                            processed_dict[field] = int(num_val)
                        else:
                            processed_dict[field] = num_val
                except (ValueError, TypeError):
                    processed_dict[field] = defaults[field]
            # Если значение уже число, оставляем как есть
            elif isinstance(value, (int, float)):
                continue
            else:
                # Если какой-то другой тип, используем значение по умолчанию
                processed_dict[field] = defaults[field]
    
    # =============================================
    # 2. СОЗДАНИЕ И ПОДГОТОВКА DATAFRAME
    # =============================================
    
    # Создаём DataFrame
    df = pd.DataFrame([processed_dict])
    
    # В демо-режиме упрощаем - создаём простые числовые признаки
    # Нормализуем числовые значения вручную
    
    normalized_features = []
    
    # Нормализуем каждое числовое поле
    for field in numeric_fields:
        if field in df.columns:
            val = df[field].iloc[0]
            # Применяем нормализацию вручную
            if field == 'age':
                normalized_features.append(val / 100.0)  # 0-1
            elif field == 'fnlwgt':
                normalized_features.append(min(val / 300000.0, 1.0))
            elif field == 'education-num':
                normalized_features.append(val / 20.0)
            elif field == 'capital-gain':
                normalized_features.append(min(val / 50000.0, 1.0))
            elif field == 'capital-loss':
                normalized_features.append(min(val / 5000.0, 1.0))
            elif field == 'hours-per-week':
                normalized_features.append(val / 80.0)
        else:
            normalized_features.append(0.5)  # значение по умолчанию
    
    # =============================================
    # 3. ОБРАБОТКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ (упрощённая)
    # =============================================
    
    # В демо-режиме просто добавляем фиктивные признаки
    # вместо реального one-hot encoding
    
    categorical_features = features_info.get('categorical_features', [])
    
    # Добавляем по 2 фиктивных признака на каждую категориальную переменную
    for cat_feature in categorical_features[:5]:  # ограничимся 5 признаками
        if cat_feature in df.columns:
            # Преобразуем категорию в число (простая хэш-функция)
            cat_value = str(df[cat_feature].iloc[0])
            hash_val = sum(ord(char) for char in cat_value) % 100 / 100.0
            normalized_features.append(hash_val)
        else:
            normalized_features.append(0.3)  # значение по умолчанию
    
    # Добиваем до 15 признаков (стандартный размер для демо)
    while len(normalized_features) < 15:
        normalized_features.append(0.0)
    
    # =============================================
    # 4. ВОЗВРАЩАЕМ РЕЗУЛЬТАТ
    # =============================================
    
    # Преобразуем в массив правильной формы: (1, количество_признаков)
    final_array = np.array([normalized_features[:15]], dtype=float)
    
    return final_array

# =============================================
# ЗАГРУЗКА РЕСУРСОВ
# =============================================
resources = load_demo_resources()

# =============================================
# ЗАГОЛОВОК
# =============================================
st.title("💰 Прогнозирование Годового Дохода")
st.markdown("""
**Предсказание, превысит ли годовой доход человека порог $50,000**

*Демо-версия приложения с использованием синтетической модели*
""")

# =============================================
# БОКОВАЯ ПАНЕЛЬ
# =============================================
with st.sidebar:
    st.header("📊 Информация о системе")
    
    if resources['loaded']:
        st.success("✅ Демо-ресурсы созданы")
        st.info("🟡 Работает в демо-режиме")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Признаков", len(resources['features_info']['numeric_features']) + 
                     len(resources['features_info']['categorical_features']))
        with col2:
            st.metric("Модель", "Demo Model")
        
        with st.expander("ℹ️ Подробности"):
            st.write("**Числовые признаки:**")
            for feat in resources['features_info']['numeric_features']:
                st.write(f"• {feat}")
            
            st.write("**Категориальные признаки:**")
            for feat in resources['features_info']['categorical_features']:
                st.write(f"• {feat}")
    else:
        st.error("⚠️ Ресурсы не созданы")
        st.write(resources['message'])
    
    st.markdown("---")
    st.caption("Версия 2.0 • Демо-режим")

# =============================================
# ОСНОВНОЕ ПРИЛОЖЕНИЕ
# =============================================
if not resources['loaded']:
    st.error("Не удалось создать демо-ресурсы. Пожалуйста, проверьте код.")
    st.stop()

# Получаем ресурсы
model = resources['model']
scaler = resources['scaler']
encoder = resources['encoder']
features_info = resources['features_info']
cat_options_ru = resources['categorical_options_ru']

# Создаем вкладки
tab1, tab2 = st.tabs(["🎯 Прогноз", "📈 Информация"])

with tab1:
    st.header("🎯 Введите параметры для прогноза")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Демография")
        age = st.slider("Возраст", 17, 90, 35)
        
        if 'sex' in cat_options_ru:
            sex = st.selectbox("Пол", cat_options_ru['sex'])
        else:
            sex = st.selectbox("Пол", ['Мужской', 'Женский'])
        
        if 'race' in cat_options_ru:
            race = st.selectbox("Раса", cat_options_ru['race'])
        else:
            race = st.selectbox("Раса", ['Белый', 'Черный', 'Другой'])
    
    with col2:
        st.subheader("🎓 Образование и Работа")
        
        if 'education' in cat_options_ru:
            education = st.selectbox("Образование", cat_options_ru['education'])
        else:
            education = st.selectbox("Образование", ['Бакалавр', 'Магистр', 'Выпускник школы'])
        
        if 'occupation' in cat_options_ru:
            occupation = st.selectbox("Профессия", cat_options_ru['occupation'])
        else:
            occupation = st.selectbox("Профессия", ['Управленческий', 'Профессиональная специализация'])
        
        if 'workclass' in cat_options_ru:
            workclass = st.selectbox("Рабочий класс", cat_options_ru['workclass'])
        else:
            workclass = st.selectbox("Рабочий класс", ['Частный', 'Федеральное правительство'])
        
        hours_per_week = st.slider("Часов в неделю", 1, 99, 40)
    
    with col3:
        st.subheader("💼 Семья и Финансы")
        
        if 'marital-status' in cat_options_ru:
            marital_status = st.selectbox("Семейное положение", cat_options_ru['marital-status'])
        else:
            marital_status = st.selectbox("Семейное положение", ['Никогда не женат/замужем', 'Женат/замужем'])
        
        if 'relationship' in cat_options_ru:
            relationship = st.selectbox("Родственные отношения", cat_options_ru['relationship'])
        else:
            relationship = st.selectbox("Родственные отношения", ['Не в семье', 'Муж', 'Жена'])
        
        capital_gain = st.number_input("Прирост капитала ($)", 0, 100000, 0)
        capital_loss = st.number_input("Потери капитала ($)", 0, 5000, 0)
    
    # Дополнительные параметры
    st.subheader("📊 Дополнительные параметры")
    col4, col5 = st.columns(2)
    
    with col4:
        fnlwgt = st.number_input("Вес наблюдения (fnlwgt)", 
                                min_value=10000, 
                                max_value=500000, 
                                value=189154)
    
    with col5:
        education_num = st.slider("Годы образования (education-num)", 1, 16, 9)
    
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
                prepared_data = prepare_demo_input(input_data, features_info, encoder, scaler)
                
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
    st.header("📈 Информация о демо-режиме")
    
    st.info("""
    **Это демо-версия приложения для прогнозирования дохода.**
    
    Функции:
    - 🎯 **Интерактивный ввод параметров** с русским интерфейсом
    - 📊 **Визуализация результатов** с вероятностями
    - 🔍 **Детальный анализ** влияния факторов
    - 💾 **Автоматическое создание модели** без загрузки файлов
    
    **Как это работает:**
    1. Вы вводите параметры человека
    2. Демо-модель анализирует введенные данные
    3. На основе простых правил (возраст, образование) вычисляется вероятность
    4. Результаты отображаются в наглядном виде
    
    **Технологии:**
    - Python + Streamlit для интерфейса
    - Scikit-learn для обработки данных
    - Pandas + NumPy для вычислений
    - Matplotlib для визуализации
    """)
    
    # Примеры
    st.subheader("📋 Примеры прогнозов")
    
    examples = [
        {
            "Описание": "👨‍💼 Успешный менеджер",
            "Возраст": 45,
            "Образование": "Магистр",
            "Профессия": "Управленческий",
            "Часы": 55,
            "Прогноз модели": "> $50K",
            "Вероятность": "85%"
        },
        {
            "Описание": "👩‍🎓 Молодой специалист",
            "Возраст": 25,
            "Образование": "Бакалавр",
            "Профессия": "Административно-канцелярский",
            "Часы": 35,
            "Прогноз модели": "≤ $50K",
            "Вероятность": "65%"
        }
    ]
    
    for example in examples:
        with st.expander(example["Описание"]):
            st.write(f"**Возраст:** {example['Возраст']} лет")
            st.write(f"**Образование:** {example['Образование']}")
            st.write(f"**Профессия:** {example['Профессия']}")
            st.write(f"**Часы работы:** {example['Часы']} ч/неделю")
            
            if example['Прогноз модели'] == "> $50K":
                st.success(f"**Прогноз:** {example['Прогноз модели']} (вероятность: {example['Вероятность']})")
            else:
                st.info(f"**Прогноз:** {example['Прогноз модели']} (вероятность: {example['Вероятность']})")

# =============================================
# ФУТЕР
# =============================================
st.markdown("---")
st.caption("📊 Демо-версия приложения для прогнозирования дохода • Streamlit + Scikit-learn")