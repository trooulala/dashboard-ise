import io
import streamlit as st
import pandas as pd

st.title('Loan Applicant Data for Credit Risk Analysis')
st.header(""" Created by Tim Karupuak Jangek """)
st.write("""M. Sulthon Sayid A""")
st.write("""Yunolva Anis R""")
st.write("""M Aziz Al Adro J""")

df = pd.read_csv("https://raw.githubusercontent.com/trooulala/dataset3-ise/main/credit_risk.csv")

st.subheader("Menampilkan 5 Data Pertama")
st.write(df.head())

df.drop(columns = 'Id', axis =1, inplace=True)
st.write("Dataframe tanpa Id, karena Id tidak relevan dengan dataset yang lainnya")
st.write(df.head())

# Membuat string dengan informasi DataFrame
buffer = io.StringIO()
df.info(buf=buffer)
info_string = buffer.getvalue()

# Menampilkan informasi DataFrame di Streamlit
st.write("Informasi DataFrame:")
st.text(info_string)

