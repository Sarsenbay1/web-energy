import datetime as dt
import pickle

import pandas as pd
import streamlit as st


with open('models/XGBRegressor.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title='Predict', page_icon='🎰')

st.write('__Введите данные для предсказания 👇__')

d = pd.Timestamp(st.date_input('__Введите дату__'))
t = st.time_input('__Введите время__')

inp = {
    'час': t.hour,
    'день недели': (d.dayofweek + 2) % 7,
    'квартал': d.quarter,
    'месяц': d.month,
    'год': d.year,
    'день в году': d.dayofyear,
    'день месяца': d.day
}
df_inp = pd.DataFrame([inp])

st.write(
    '## Для следующего набора данных:', 
    df_inp, 
    '## Получаем следующее предсказание:'
    )

st.write(f'### Расчетное энергопотребление в мегаваттах (МВт): {model.predict(df_inp)} ⚡')
