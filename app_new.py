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

@st.cache_resource
def load_resources():
    """Загрузка или создание всех необходимых ресурсов. Создает демо-модель на лету."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    import pandas as pd
    
    resources = {
        'model': None,
        'scaler': None,
        'encoder': None,
        'features_info': None,
        'categorical_options': None,
        'loaded': False,
        'message': '',
        'demo_mode': True  # Флаг, что работает демо-режим
    }
    
    try:
        # --- 1. СОЗДАЁМ ДЕМО-МОДЕЛЬ (основная логика) ---
        class IncomeDemoModel:
            """Простая модель для демонстрации логики приложения."""
            def predict(self, X):
                # X - это уже подготовленные данные (масштабированные, закодированные)
                # Для демо просто возвращаем случайные 0 или 1
                np.random.seed(42) # Для воспроизводимости
                return np.random.randint(0, 2, X.shape[0])
            
            def predict_proba(self, X):
                # Возвращаем "вероятности". Для реалистичности сделаем их зависимыми от первого признака (age)
                np.random.seed(42)
                n_samples = X.shape[0]
                # Базовый шанс высокого дохода - 30%
                base_prob = 0.3
                # Немного увеличим шанс, если "возраст" (первый столбец в масштабированных данных) высокий
                if n_samples > 0 and X.shape[1] > 0:
                    # Предполагаем, что age - первый признак. Нормализуем его влияние.
                    age_effect = X[:, 0] * 0.1 if X.shape[1] > 0 else 0
                    age_effect = np.clip(age_effect, -0.3, 0.3)
                    base_prob += age_effect
                
                prob_high = np.clip(base_prob + np.random.normal(0, 0.1, n_samples), 0.05, 0.95)
                prob_high = prob_high.reshape(-1, 1)
                return np.hstack([1 - prob_high, prob_high])
        
        # --- 2. СОЗДАЁМ И "ОБУЧАЕМ" ПРЕПРОЦЕССОРЫ ---
        # Создаём скейлер (просто пустой, для совместимости интерфейса)
        demo_scaler = StandardScaler()
        # Для работы .transform() скейлеру нужно быть "обученным" на каких-то данных.
        # Создадим синтетические данные для "обучения".
        dummy_numeric_data = np.array([[30, 200000, 9, 0, 0, 40]])  # [age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week]
        demo_scaler.fit(dummy_numeric_data)
        
        # Создаём энкодер
        demo_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Создаём синтетические категориальные данные для "обучения" энкодера
        dummy_cat_data = pd.DataFrame({
            'workclass': ['Private'],
            'education': ['HS-grad'],
            'marital-status': ['Never-married'],
            'occupation': ['Prof-specialty'],
            'relationship': ['Not-in-family'],
            'race': ['White'],
            'sex': ['Male']
        })
        demo_encoder.fit(dummy_cat_data)
        
        # --- 3. ЗАПОЛНЯЕМ МЕТАДАННЫЕ (точно как в интерфейсе) ---
        resources['features_info'] = {
            'numeric_features': ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'],
            'categorical_features': ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex'],
            'all_features': []  # Можно оставить пустым или сгенерировать
        }
        
        resources['categorical_options'] = {
            'sex': ['Male', 'Female'],
            'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
            'education': ['Bachelors', 'Some-college', 'HS-grad', 'Masters', 'Assoc-voc', 'Assoc-acdm', '11th', '9th', '7th-8th', '12th', '10th', 'Doctorate', '5th-6th', '1st-4th', 'Preschool'],
            'marital-status': ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
            'relationship': ['Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
            'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay'],
            'occupation': ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
        }
        
        # --- 4. СОЗДАЁМ РУССКИЕ ВЕРСИИ КАТЕГОРИЙ (categorical_options_ru) ---
        # Словарь перевода (убедитесь, что он объявлен глобально в вашем файле)
        TRANSLATION_DICT = {
            'sex': {'Male': 'Мужской', 'Female': 'Женский'},
            'race': {'White': 'Белый', 'Black': 'Черный', 'Asian-Pac-Islander': 'Азиатско-Тихоокеанский островитянин', 'Amer-Indian-Eskimo': 'Индеец/Эскимос', 'Other': 'Другой'},
            'education': {'Bachelors': 'Бакалавр', 'Some-college': 'Неоконченное высшее', 'HS-grad': 'Выпускник школы', 'Masters': 'Магистр', 'Assoc-voc': 'Профессионально-техническое', 'Assoc-acdm': 'Академическое (2 года)', '11th': '11 класс', '9th': '9 класс', '7th-8th': '7-8 классы', '12th': '12 класс', '10th': '10 класс', 'Doctorate': 'Доктор', '5th-6th': '5-6 классы', '1st-4th': '1-4 классы', 'Preschool': 'Дошкольное'},
            'marital-status': {'Never-married': 'Никогда не женат/замужем', 'Married-civ-spouse': 'Женат/замужем (гражданский брак)', 'Divorced': 'В разводе', 'Married-spouse-absent': 'Женат/замужем (супруг отсутствует)', 'Separated': 'Разведен/разведена', 'Married-AF-spouse': 'Женат/замужем (военнослужащий)', 'Widowed': 'Вдовец/вдова'},
            'relationship': {'Not-in-family': 'Не в семье', 'Husband': 'Муж', 'Wife': 'Жена', 'Own-child': 'Собственный ребенок', 'Unmarried': 'Не женат/не замужем', 'Other-relative': 'Другой родственник'},
            'workclass': {'Private': 'Частный', 'Self-emp-not-inc': 'Самостоятельный (не инкорпорированный)', 'Self-emp-inc': 'Самостоятельный (инкорпорированный)', 'Federal-gov': 'Федеральное правительство', 'Local-gov': 'Местное правительство', 'State-gov': 'Правительство штата', 'Without-pay': 'Без оплаты'},
            'occupation': {'Prof-specialty': 'Профессиональная специализация', 'Craft-repair': 'Ремесло-ремонт', 'Exec-managerial': 'Управленческий', 'Adm-clerical': 'Административно-канцелярский', 'Sales': 'Продажи', 'Other-service': 'Другие услуги', 'Machine-op-inspct': 'Машинные операторы-инспекторы', 'Transport-moving': 'Транспортировка-переезд', 'Handlers-cleaners': 'Грузчики-уборщики', 'Farming-fishing': 'Сельское хозяйство-рыболовство', 'Tech-support': 'Техподдержка', 'Protective-serv': 'Охранные услуги', 'Priv-house-serv': 'Частные домашние услуги', 'Armed-Forces': 'Вооруженные силы'}
        }
        
        resources['categorical_options_ru'] = {}
        for category, eng_values in resources['categorical_options'].items():
            if category in TRANSLATION_DICT:
                rus_values = [TRANSLATION_DICT[category].get(val, val) for val in eng_values]
                resources['categorical_options_ru'][category] = rus_values
            else:
                resources['categorical_options_ru'][category] = eng_values
        
        # --- 5. ПРИСВАИВАЕМ СОЗДАННЫЕ ОБЪЕКТЫ ---
        resources['model'] = IncomeDemoModel()
        resources['scaler'] = demo_scaler
        resources['encoder'] = demo_encoder
        resources['loaded'] = True
        resources['message'] = "✅ Все ресурсы созданы в демо-режиме. Приложение готово к работе.\n"
        
    except Exception as e:
        resources['message'] = f"❌ Критическая ошибка при создании демо-ресурсов: {str(e)}"
    
    return resources