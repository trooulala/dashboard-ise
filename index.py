import io
from matplotlib import pyplot as plt
from sklearn import tree
import streamlit as st
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

st.title('Loan Applicant Data for Credit Risk Analysis')
st.header(""" Created by Tim Karupuak Jangek """)
st.write("""M. Sulthon Sayid A""")
st.write("""Yunolva Anis R""")
st.write("""M Aziz Al Adro J""")

df = pd.read_csv("https://raw.githubusercontent.com/trooulala/dataset3-ise/main/credit_risk.csv")

st.subheader("Menampilkan 5 Data Pertama")
st.write(df.head())

st.subheader("Feature Engineering")

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

# Membuat imputer
imputer = SimpleImputer(strategy='mean')

# Mengisi kolom 'Emp_length' dan 'Rate' dengan mean
df[['Emp_length', 'Rate']] = imputer.fit_transform(df[['Emp_length', 'Rate']])

# Membuat string dengan informasi DataFrame setelah pengisian
buffer_after = io.StringIO()
df.info(buf=buffer_after)
info_string_after = buffer_after.getvalue()

st.write("Informasi DataFrame Setelah Pengisian:")
st.text(info_string_after)

df['Home'] = df['Home'].factorize()[0]

st.write("DataFrame setelah faktorisasi kolom 'Home':")
st.write(df.head())

df['Intent'] = df['Intent'].factorize()[0]

st.write("DataFrame setelah faktorisasi kolom 'Intent':")
st.write(df.head())

df['Default'] = df['Default'].factorize()[0]

st.write("DataFrame setelah faktorisasi kolom 'Default':")
st.write(df.head())

# Memisahkan fitur dan target
X = df.drop(['Status'], axis=1)
y = df['Status']
st.write("Menampilkan Kolom 'Status' sebagai target")
st.write(y)

# Menampilkan DataFrame X (fitur)
st.write("DataFrame X (fitur) setelah menghapus kolom 'Status':")
st.write(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Menampilkan ukuran dataset
st.write("Ukuran Dataset Pelatihan (X_train, y_train):")
st.write(f"Jumlah Baris: {X_train.shape[0]}, Jumlah Kolom: {X_train.shape[1]}")

st.write("Ukuran Dataset Pengujian (X_test, y_test):")
st.write(f"Jumlah Baris: {X_test.shape[0]}, Jumlah Kolom: {X_test.shape[1]}")

# Standard Scaling
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Label Encoding pada kolom target
le = LabelEncoder()
le.fit(y_train)

# Mengambil kategori unik setelah Label Encoding
classes = le.classes_

# Menampilkan kategori unik di Streamlit
st.write("Kategori Unik Setelah Label Encoding:")
st.write(classes)

model_dt = DecisionTreeClassifier(max_depth=6, random_state=10)
model_dt.fit(X_train_scaled, y_train)

def performance_check(clf, X, y, classes):
  y_pred = clf.predict(X)
  cm = confusion_matrix(y, y_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
  disp.plot()
  plt.title("Confusion Matrix")
  st.pyplot(plt.gcf())  # Menggunakan plt.gcf() untuk mendapatkan figur saat ini

  report = classification_report(y, y_pred, target_names=classes)
  st.write("Classification Report:")
  st.write(report)

classes = ["yes", "no"]

st.write('Decision Tree - Train')
performance_check(model_dt, X_train_scaled, y_train, classes)

st.write('Decision Tree - Test')
performance_check(model_dt, X_test_scaled, y_test, classes)

# Menampilkan visualisasi pohon keputusan
fig, ax = plt.subplots(figsize=(20, 10))
tree.plot_tree(model_dt,
               feature_names=X.columns.tolist(),
               class_names=["yes", "no"],
               filled=True,
               fontsize=20,
               max_depth=2)
st.pyplot(fig)


st.subheader("Hypertuning Paramter")
# Membuat model Decision Tree
model = DecisionTreeClassifier(random_state=10)

param_grid = {
    'max_depth': [2, 4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Melakukan Grid Search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Menampilkan parameter terbaik
st.write("Parameter terbaik:")
st.write(grid_search.best_params_)

# Mendapatkan model terbaik
best_model = grid_search.best_estimator_

# Menampilkan performa model terbaik
st.write('Decision Tree - Test (Model Terbaik)')
performance_check(best_model, X_test_scaled, y_test, classes)


# Melakukan prediksi dengan model terbaik
y_pred = best_model.predict(X_test_scaled)

# Mendapatkan laporan performa
report = classification_report(y_test, y_pred, target_names=classes)

st.write("Performance Report:")
st.write(report)