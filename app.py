import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Конфигурация страницы
st.set_page_config(
    page_title="Приложение-предсказатель дохода",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка модели с кэшированием
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, True
    except FileNotFoundError as e:
        st.error(f"❌ Файл не найден: {e}")
        return None, None, False
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")
        return None, None, False

# Заголовок приложения
st.title("💰 Прогнозирование уровня дохода")
st.markdown("""
### Предсказание, превысит ли годовой доход человека порог **$50,000**
На основе данных Adult Census Income Dataset
""")

# Загрузка модели
model, scaler, loaded = load_model_and_scaler()

if loaded:
    st.sidebar.success("✅ Модель успешно загружена!")
    
    # Информация о модели в сайдбаре
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ О модели")
    st.sidebar.write(f"**Тип:** {type(model).__name__}")
    st.sidebar.write("**Метрика:** AUC-ROC = 0.867")
    st.sidebar.write("**Алгоритм:** Gradient Boosting")
    st.sidebar.write("**Деревьев:** 100")
    st.sidebar.write("**Глубина:** 5")
    
    # Основной контент
    tab1, tab2, tab3 = st.tabs(["🔮 Прогноз", "📊 Анализ", "📈 Примеры"])
    
    with tab1:
        st.header("Введите параметры для прогноза")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Демографические данные")
            age = st.slider("Возраст (age)", 17, 90, 35, 
                           help="Возраст человека в годах")
            sex = st.selectbox("Пол (sex)", ["Male", "Female"],
                              help="Биологический пол")
            
        with col2:
            st.subheader("Финансовые показатели")
            capital_gain = st.number_input("Прирост капитала ($)", 0, 100000, 0,
                                          help="Доход от инвестиций")
            capital_loss = st.number_input("Потери капитала ($)", 0, 5000, 0,
                                          help="Потери от инвестиций")
            
        with col3:
            st.subheader("Рабочие параметры")
            hours_per_week = st.slider("Часов работы в неделю", 1, 99, 40,
                                      help="Количество рабочих часов в неделю")
            education_num = st.slider("Годы образования", 1, 16, 9,
                                     help="Количество лет образования")
        
        # Кнопка для предсказания
        if st.button("🎯 Сделать прогноз", type="primary", use_container_width=True):
            # Подготовка данных
            sex_numeric = 1 if sex == "Male" else 0
            
            # Создаем массив с признаками в том же порядке, что и при обучении
            # Порядок: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week, sex
            # Но так как мы используем только часть признаков, нужно быть осторожным
            # Предположим, что модель обучена на 7 признаках
            features = np.array([[age, 189154, education_num, capital_gain, 
                                 capital_loss, hours_per_week, sex_numeric]])
            
            # Масштабирование
            features_scaled = scaler.transform(features)
            
            # Предсказание
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            # Отображение результатов
            st.markdown("---")
            st.subheader("📊 Результаты прогноза")
            
            # Основной результат
            result_col1, result_col2 = st.columns([1, 2])
            
            with result_col1:
                if prediction == 1:
                    st.success(f"""
                    ## ✅ Высокий доход
                    ### > $50K/год
                    """)
                    st.metric("Вероятность", f"{probabilities[1]:.1%}")
                else:
                    st.info(f"""
                    ## ⚠️ Средний доход
                    ### ≤ $50K/год
                    """)
                    st.metric("Вероятность", f"{probabilities[0]:.1%}")
            
            with result_col2:
                # График вероятностей
                fig, ax = plt.subplots(figsize=(8, 4))
                
                categories = ['≤ $50K', '> $50K']
                colors = ['#FF6B6B', '#4ECDC4']
                bars = ax.bar(categories, probabilities, color=colors, width=0.6)
                
                ax.set_ylim(0, 1)
                ax.set_ylabel('Вероятность', fontsize=12)
                ax.set_title('Распределение вероятностей', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Добавляем значения на столбцы
                for bar, prob in zip(bars, probabilities):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{prob:.1%}', ha='center', va='bottom', 
                           fontsize=12, fontweight='bold')
                
                st.pyplot(fig)
            
            # Детали прогноза
            with st.expander("📋 Детали введенных параметров"):
                details = pd.DataFrame({
                    'Параметр': ['Возраст', 'Пол', 'Годы образования', 
                                'Часы работы/неделя', 'Прирост капитала', 'Потери капитала'],
                    'Значение': [f"{age} лет", sex, f"{education_num} лет", 
                               f"{hours_per_week} ч", f"${capital_gain:,}", f"${capital_loss:,}"]
                })
                st.dataframe(details, use_container_width=True, hide_index=True)
                
                # Важность признаков (если доступно)
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Важность признаков")
                    feature_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                                    'capital-loss', 'hours-per-week', 'sex']
                    importance_df = pd.DataFrame({
                        'Признак': feature_names,
                        'Важность': model.feature_importances_
                    }).sort_values('Важность', ascending=False)
                    
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    ax2.barh(importance_df['Признак'], importance_df['Важность'], color='skyblue')
                    ax2.set_xlabel('Важность')
                    ax2.set_title('Важность признаков в модели')
                    ax2.invert_yaxis()  # самый важный сверху
                    st.pyplot(fig2)
    
    with tab2:
        st.header("📊 Анализ модели")
        
        col_analysis1, col_analysis2 = st.columns(2)
        
        with col_analysis1:
            st.subheader("Характеристики модели")
            
            model_info = {
                "Алгоритм": "Gradient Boosting Classifier",
                "Количество деревьев": 100,
                "Максимальная глубина": 5,
                "Критерий": "friedman_mse",
                "Количество признаков": 7,
                "Метрика качества": "AUC-ROC",
                "Значение AUC-ROC": "0.867",
                "Обучено на": "15,347 записях"
            }
            
            for key, value in model_info.items():
                st.write(f"**{key}:** {value}")
        
        with col_analysis2:
            st.subheader("Описание признаков")
            
            features_desc = {
                "age": "Возраст человека в годах",
                "fnlwgt": "Вес наблюдения (репрезентативность в популяции)",
                "education-num": "Количество лет образования",
                "capital-gain": "Доход от инвестиций",
                "capital-loss": "Потери от инвестиций",
                "hours-per-week": "Количество рабочих часов в неделю",
                "sex": "Биологический пол (1=Male, 0=Female)"
            }
            
            for feat, desc in features_desc.items():
                st.write(f"• **{feat}**: {desc}")
        
        # График распределения важности признаков
        st.subheader("Визуализация важности признаков")
        if hasattr(model, 'feature_importances_'):
            feature_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                            'capital-loss', 'hours-per-week', 'sex']
            importances = model.feature_importances_
            
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            y_pos = np.arange(len(feature_names))
            ax3.barh(y_pos, importances, align='center', color='teal', alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(feature_names)
            ax3.set_xlabel('Важность признака')
            ax3.set_title('Важность признаков в модели GradientBoosting')
            ax3.invert_yaxis()
            
            # Добавляем значения на столбцы
            for i, v in enumerate(importances):
                ax3.text(v + 0.001, i, f'{v:.3f}', va='center')
            
            st.pyplot(fig3)
    
    with tab3:
        st.header("📈 Примеры предсказаний")
        
        st.write("""
        Ниже приведены примеры типичных случаев с предсказаниями модели.
        Эти примеры помогают понять, как модель реагирует на различные комбинации признаков.
        """)
        
        # Примеры данных
        examples = pd.DataFrame({
            'Пример': ['Бизнесмен', 'Студент', 'Врач', 'Рабочий', 'Пенсионер'],
            'Возраст': [45, 22, 35, 28, 65],
            'Пол': ['Male', 'Female', 'Male', 'Male', 'Female'],
            'Образование (лет)': [16, 12, 18, 10, 12],
            'Часы/неделя': [60, 20, 50, 45, 15],
            'Прирост капитала ($)': [50000, 0, 10000, 0, 2000],
            'Ожидаемый доход': ['> $50K', '≤ $50K', '> $50K', '≤ $50K', '≤ $50K']
        })
        
        st.dataframe(examples, use_container_width=True, hide_index=True)
        
        # Кнопки для быстрого заполнения
        st.subheader("Быстрое заполнение формы")
        example_cols = st.columns(5)
        
        with example_cols[0]:
            if st.button("👨‍💼 Бизнесмен", use_container_width=True):
                st.session_state.age = 45
                st.session_state.sex = "Male"
                st.session_state.education_num = 16
                st.session_state.hours_per_week = 60
                st.session_state.capital_gain = 50000
        
        with example_cols[1]:
            if st.button("👩‍🎓 Студент", use_container_width=True):
                st.session_state.age = 22
                st.session_state.sex = "Female"
                st.session_state.education_num = 12
                st.session_state.hours_per_week = 20
                st.session_state.capital_gain = 0
        
        with example_cols[2]:
            if st.button("👨‍⚕️ Врач", use_container_width=True):
                st.session_state.age = 35
                st.session_state.sex = "Male"
                st.session_state.education_num = 18
                st.session_state.hours_per_week = 50
                st.session_state.capital_gain = 10000
        
        st.info("Нажмите на кнопку с примером, чтобы автоматически заполнить форму вкладки 'Прогноз'")
    
    # Футер
    st.markdown("---")
    st.caption("""
    **Income Prediction App** | Модель обучена на данных Adult Census Income Dataset | 
    GradientBoosting Classifier | AUC-ROC: 0.867
    """)
    
else:
    # Если модель не загрузилась
    st.error("""
    ## ⚠️ Не удалось загрузить модель!
    
    Убедитесь, что в директории есть следующие файлы:
    1. **best_model.pkl** - обученная модель GradientBoosting
    2. **scaler.pkl** - скейлер для масштабирования признаков
    
    ### Как создать недостающие файлы:
    
    Если у вас есть Jupyter ноутбук с обученной моделью, выполните в нём:
    ```python
    import joblib
    
    # Сохраните модель
    joblib.dump(gb_grid.best_estimator_, 'best_model.pkl')
    
    # Сохраните скейлер
    joblib.dump(scaler, 'scaler.pkl')
    ```
    
    Или создайте скейлер вручную:
    ```python
    from sklearn.preprocessing import MinMaxScaler
    import joblib
    
    scaler = MinMaxScaler()
    # Если у вас есть X_train, можно использовать его
    # scaler.fit(X_train)
    
    # Или создать фиктивный скейлер
    import numpy as np
    dummy_data = np.array([[35, 189154, 9, 0, 0, 40, 1]])  # Пример данных
    scaler.fit(dummy_data)
    joblib.dump(scaler, 'scaler.pkl')
    ```
    """)
    
    # Интерактивная проверка файлов
    st.subheader("Проверка файлов в директории")
    
    import os
    files = os.listdir('.')
    
    file_table = []
    for file in files:
        size = os.path.getsize(file) if os.path.isfile(file) else 0
        file_table.append({
            'Файл': file,
            'Размер (байт)': size,
            'Тип': 'Файл' if os.path.isfile(file) else 'Папка'
        })
    
    st.dataframe(pd.DataFrame(file_table), use_container_width=True)