import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import google.generativeai as genai
from sklearn.ensemble import RandomForestRegressor
# === Page Config ===
st.set_page_config(
    page_title="🚘 Car Assistant",
    page_icon="🧠",
    layout="wide"
)
# === THEME SWITCHER ===
theme = st.radio("🌗 Choose Theme", ["Light mode", "Dark mode"], horizontal=True)

if theme == "Light mode":
    background_color = "#f5f0e6"
    text_color = "#1f1f1f"
    button_color = "#e0dbd1"
    hover_color = "#d1cfc7"
else:
    background_color = "#1e1e1e"
    text_color = "#808000"
    button_color = "#333333"
    hover_color = "#444444"

# === Dynamic CSS based on theme ===
st.markdown(f""" <style>
html, body, [class*="css"], .main, .block-container {{
    background-color: {background_color} !important;
    color: {text_color} !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}}
h1, h2, h3, h4, h5, h6, p, span, label, div {{
    color: {text_color} !important;
    font-weight: 600 !important;
}}
.stButton > button {{
    background-color: {button_color};
    color: {text_color};
    border: 1px solid #888;
    padding: 0.5rem 1rem;
    border-radius: 12px;
    transition: all 0.2s ease-in-out;
    font-size: 16px;
    font-weight: 600;
}}
.stButton > button:hover {{
    background-color: {hover_color};
    transform: scale(1.05);
}}
input, textarea, select {{
    background-color: {button_color} !important;
    color: {text_color} !important;
    border: 1px solid #aaa !important;
    border-radius: 8px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}}
::placeholder {{
    color: {text_color}99 !important;
    font-size: 15px !important;
}} </style>
""", unsafe_allow_html=True)

# === AI Подбор Gemini ===
genai.configure(api_key="AIzaSyAv1xsDITJNVKMtPSVN8gQfYgOWcIPBoDo")
model = genai.GenerativeModel("models/gemini-1.5-flash")

body_types = {
    'пикап': "Пикап — это мощный автомобиль с открытым грузовым отсеком...",
    'минивэн': "Минивэн — это просторный автомобиль...",
    'кроссовер': "Кроссовер — это универсальный автомобиль...",
    'седан': "Седан — классический тип автомобиля...",
    'хэтчбек': "Хэтчбек — компактный автомобиль...",
    'универсал': "Универсал — автомобиль с удлинённым кузовом...",
    'внедорожник': "Внедорожник — крупный автомобиль с высокой проходимостью..."
}

transmissions = {
    'автомат': "Автоматическая коробка передач обеспечивает плавную езду...",
    'механика': "Механическая коробка передач даёт полный контроль..."
}

fuels = {
    'бензин': "Бензиновые двигатели обеспечивают хорошую динамику...",
    'дизель': "Дизельные двигатели отличаются высокой экономичностью...",
    'электро': "Электромобили полностью работают на электричестве...",
    'гибрид': "Гибридные автомобили совмещают бензиновый двигатель и электромотор..."
}

# === Odometer Checker ===
@st.cache_data
def load_data():
    df = pd.read_csv("22613data.csv")
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def preprocess_odometer_model(df):
    df = df.copy()
    df = df.dropna(subset=["Year", "Volume", "Mileage", "Mark"])
    group_median = (
        df.groupby(["Year", "Mark", "Volume"])["Mileage"]
        .median()
        .reset_index()
        .rename(columns={"Mileage": "MedianMileage"})
    )
    df = df.merge(group_median, on=["Year", "Mark", "Volume"], how="left")
    df["OdometerNormal"] = (df["Mileage"] >= 0.65 * df["MedianMileage"]).astype(int)
    cat_cols = ["Mark", "Fuel Type", "Transmission", "Car_type"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    features = ["Year", "Volume", "Mileage", "Mark", "Fuel Type", "Transmission", "Car_type"]
    X = df[features]
    y = df["OdometerNormal"]
    return X, y, le_dict

@st.cache_data
def train_odometer_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
@st.cache_data
def train_price_model(df,_encoders):
    df = df.copy()
    df = df.dropna(subset=["Price", "Year", "Mileage"])
    df = df[df["Price"] > 100000]

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.upper()
        df[col] = encoders[col].transform(df[col])

    X = df[["Company", "Mark", "Year", "Fuel Type", "Transmission", "Mileage", "Car_type"]]
    y = df["Price"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# === Interface ===
st.title("🚘 Car Assistant")

# Загрузка данных
raw_data = load_data()

# Классификаторы для категориальных признаков
categorical_cols = ["Company", "Mark", "Fuel Type", "Transmission", "Car_type"]
encoders = {col: LabelEncoder().fit(raw_data[col].astype(str).str.upper()) for col in categorical_cols}

# Тabs
tabs = st.tabs(["AI подбор", "Проверка пробега", "Оценка цены", "Кредитный калькулятор"])

# === Tab 1: AI подбор ===
with tabs[0]:
    st.header("🧠 Умный подбор авто")
    user_input = st.text_input("Опиши, какой автомобиль тебе нужен:")

    if user_input:
        with st.spinner("Создание рекомендации..."):
            prompt = f"""
Ты — автомобильный эксперт. На основе описаний подбираешь лучший вариант машины под запрос.
Но обязательно учитываешь, чтобы такие комплектации реально существовали — например, не предлагай электрокар с механикой.
Отвечаешь на том языке, на котором тебе задали вопрос.
"{user_input}"

Описания кузова:
{chr(10).join([f"{k} — {v}" for k, v in body_types.items()])}

Описания трансмиссии:
{chr(10).join([f"{k} — {v}" for k, v in transmissions.items()])}

Описания топлива:
{chr(10).join([f"{k} — {v}" for k, v in fuels.items()])}

Ответь:
1. Лучший тип кузова (с объяснением)
2. Лучшая трансмиссия (с объяснением)
3. Лучший тип топлива (с объяснением)
4. Общее заключение
"""
            response = model.generate_content(prompt)
            st.markdown("### 🤖 Рекомендация:")
            st.write(response.text)

# === Tab 2: Проверка пробега ===
with tabs[1]:
    df = raw_data.copy()
    st.subheader("🔍 Проверка достоверности пробега")
    X, y, le_dict = preprocess_odometer_model(df)
    model = train_odometer_model(X, y)

    st.markdown("### Введите параметры автомобиля:")
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Company", df["Company"].unique())
        mark = st.selectbox("Mark", df[df["Company"] == company]["Mark"].unique())
        year = st.number_input("Year", min_value=2000, max_value=2025, value=2015)
        mileage = st.number_input("Mileage (km)", value=100000)
    with col2:
        volume = st.number_input("Engine Volume (L)", value=2.0)
        fuel = st.selectbox("Fuel Type", df["Fuel Type"].unique())
        trans = st.selectbox("Transmission", df["Transmission"].unique())
        ctype = st.selectbox("Car Type", df["Car_type"].unique())

    if st.button("Проверить пробег"):
        input_dict = {
            "Year": year,
            "Volume": volume,
            "Mileage": mileage,
            "Mark": le_dict["Mark"].transform([mark])[0],
            "Fuel Type": le_dict["Fuel Type"].transform([fuel])[0],
            "Transmission": le_dict["Transmission"].transform([trans])[0],
            "Car_type": le_dict["Car_type"].transform([ctype])[0],
        }
        X_input = pd.DataFrame([input_dict])
        prob = model.predict_proba(X_input)[0][1]
        pred = model.predict(X_input)[0]

        if pred == 1:
            st.success("✅ Пробег выглядит достоверным")
        else:
            st.warning("⚠️ Возможна скрутка пробега")

# === Tab 3: Оценка цены ===
with tabs[2]:
    st.markdown("### 📊 Введите данные автомобиля для оценки стоимости:")
    price_model = train_price_model(raw_data, encoders)
    company = st.selectbox("🏢 Производитель", sorted(raw_data['Company'].dropna().unique()), key="company_select")
    filtered_data = raw_data[raw_data['Company'] == company]
    mark = st.selectbox("🚘 Модель", sorted(filtered_data['Mark'].dropna().unique()), key="model_select")
    year = st.number_input("📅 Год", 1990, 2025, 2015, key="year_input")
    fuel = st.selectbox("⛽ Тип топлива", sorted(raw_data['Fuel Type'].dropna().unique()), key="fuel_select")
    trans = st.selectbox("⚙️ Трансмиссия", sorted(raw_data['Transmission'].dropna().unique()), key="trans_select")
    mileage = st.number_input("🛣️ Пробег (км)", 0, 1_000_000, 100_000, key="mileage_input")
    car_type = st.selectbox("🚗 Тип кузова", sorted(raw_data['Car_type'].dropna().unique()), key="type_select")
    if st.button("📈 Оценить стоимость", key="price_button"):
        new_car = pd.DataFrame({
            'Company': [company],
            'Mark': [mark],
            'Year': [year],
            'Fuel Type': [fuel],
            'Transmission': [trans],
            'Mileage': [mileage],
            'Car_type': [car_type]
        })

        try:
            for col in categorical_cols:
                new_car[col] = new_car[col].astype(str).str.upper()
                if any(v not in encoders[col].classes_ for v in new_car[col]):
                    raise ValueError(f"❌ Unknown value in column '{col}'")
                new_car[col] = encoders[col].transform(new_car[col])

            pred_price = price_model.predict(new_car)[0]
            st.success(f"💵 Оценочная стоимость: **{int(pred_price):,} ₸**")

            filtered_similar = raw_data[
                (raw_data["Company"] == company) &
                (raw_data["Mark"] == mark) &
                (abs(raw_data["Year"] - year) <= 2) &
                (raw_data["Fuel Type"] == fuel) &
                (raw_data["Transmission"] == trans) &
                (abs(raw_data["Mileage"] - mileage) <= 40000) &
                (raw_data["Car_type"] == car_type)
            ]
            if not filtered_similar.empty:
                st.markdown("### 🔍 Похожие предложения:")
                st.dataframe(filtered_similar.head(3).reset_index(drop=True))
            else:
                st.info("Похожие предложения не найдены.")
        except ValueError as e:
            st.error(str(e))

# === Tab 4: Кредитный калькулятор ===
with tabs[3]:
    st.markdown("### 📆 Кредитный калькулятор")
    car_price = st.number_input("Цена автомобиля (₸)", min_value=100000, value=1000000, step=10000)
    down_payment = st.number_input("Первоначальный взнос (₸)", min_value=0, max_value=car_price, value=int(car_price * 0.2), step=10000)
    term = st.slider("Срок (мес)", 6, 84, 36, step=6)
    rate = st.slider("Процентная ставка (% в год)", 0.0, 100.0, 10.0, step=0.1)

    if car_price > down_payment:
        loan = car_price - down_payment
        monthly_rate = (rate / 100) / 12

        if rate > 0:
            m = monthly_rate
            monthly = loan * (m * (1 + m)**term) / ((1 + m)**term - 1)
        else:
            monthly = loan / term

        st.success(f"Ежемесячный платёж: **{int(monthly):,} ₸**")
    else:
        st.warning("Первоначальный взнос должен быть меньше цены автомобиля")
