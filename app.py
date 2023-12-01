import streamlit as st
import pandas as pd


view = [100, 150, 30]

st.write('# 유튜브 조회수')

st.write('## Raw')
view

st.write('## Bar chart')
st.bar_chart(view)

sview = pd.Series(view)
sview


