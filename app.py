import sklearn

#from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import streamlit as st
import pickle
import numpy as np

import base64
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg('5.jpg')


classifier_name=['LightGBM']
option = st.sidebar.selectbox('Алгоритм для прогнозирования:', classifier_name)
st.subheader(option)



#Importing model and label encoders
model=pickle.load(open("final_xg_model.pkl","rb"))
#model_1 = pickle.load(open("final_rf_model.pkl","rb"))
le_pik=pickle.load(open("label_encoding_for_gender.pkl","rb"))
le1_pik=pickle.load(open("label_encoding_for_geo.pkl","rb"))

def predict_churn(CreditScore, Geo, Gen, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    input = np.array([[CreditScore, Geo, Gen, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]).astype(np.float64)
    if option == 'LightGBM':
        prediction = model.predict_proba(input)
        pred = '{0:.{1}f}'.format(prediction[0][0], 2)

    else:
        pred=0.30
        #st.markdown('Клиент останется в банке')

    return float(pred)


def main():
    st.title("Прогноз оттока клиентов из банка")
    html_temp = """
    <div style="background-color:white ;padding:10px">
    <h2 style="color:red;text-align:center;">Введите данные по клиенту:</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)





    st.sidebar.subheader("Приложение создано в рамках проекта IT-Academy по направлению Data Science")
    st.sidebar.image('4.jpg')
    st.sidebar.text("Разработано Слука М.З., ЦБУ 602 г. Лида")
    st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#8993ab,#8993ab);
    color: black;
}
</style>
""",
    unsafe_allow_html=True,
)

    CreditScore = st.slider('Скоринговый балл', 0, 400)

    Geography = st.selectbox('Регион', ['France', 'Germany', 'Spain'])
    Geo = int(le1_pik.transform([Geography]))
       
    Gender = st.selectbox('Пол', ['Male', 'Female'])
    Gen = int(le_pik.transform([Gender]))

    Age = st.slider("Возраст", 18, 100)

    Tenure = st.selectbox("Продолжительность обслуживания в банке, лет", ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10','11','12','13','14','15'])

    Balance = st.slider("Баланс счёта", 0.00, 25000.00)

    NumOfProducts = st.selectbox('Количество банковских продуктов', ['1', '2', '3', '4'])

    HasCrCard = st.selectbox("Наличие кредитной карточки", ['0', '1'])

    IsActiveMember = st.selectbox("Активность", ['0', '1'])

    EstimatedSalary = st.slider("Заработная плата", 0.00, 25000.00)

                
    churn_html = """  
              <div style="background-color:#f44336;padding:20px >
               <h2 style="color:red;text-align:center;"> К сожалению, мы теряем клиента...</h2>
               </div>
            """
    no_churn_html = """  
              <div style="background-color:#94be8d;padding:20px >
               <h2 style="color:green ;text-align:center;"> Успех, клиент остаётся в банке!</h2>
               </div>
            """
            
    if int(Age)-int(Tenure)<17:
        st.error('Некорректный ввод данных по возрасту клиента и/или продолжительности обслуживания в банке')


    if st.button('Сделать прогноз'):
        output = predict_churn(CreditScore, Geo, Gen, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        st.success('Вероятность оттока составляет {}'.format(output))
        

        if output >= 0.5:
            st.markdown(churn_html, unsafe_allow_html= True)

        else:
            st.markdown(no_churn_html, unsafe_allow_html= True)

                
    if int(Age)< 18:
            st.error('Некорректный ввод данных по возрасту клиента')

    if int(Age)> 92:
            st.error('Некорректный ввод данных по возрасту клиента')

    if int(Tenure)> 10:
            st.error('Некорректный ввод данных по продолжительности обслуживания в банке')
        
    if CreditScore < 100 and Balance < 5000 and EstimatedSalary < 5000 and IsActiveMember == 0 and Age > 80 and Tenure == 0:
            st.success('Вероятность оттока составляет более 90%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore < 200 and Balance < 10000 and EstimatedSalary < 10000 and IsActiveMember == 0 and Age > 70 and Tenure == 1:
            st.success('Вероятность оттока составляет более 70%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore < 300 and Balance < 15000 and EstimatedSalary < 15000 and IsActiveMember == 0 and Age > 60 and Tenure == 2:
            st.success('Вероятность оттока составляет более 50%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore > 100 and EstimatedSalary > 5000 and IsActiveMember == 1 and NumOfProducts > 1 and Age < 60 and Tenure > 3 and Balance > 5000:
            st.success('Вероятность оттока составляет менее 30%.')
            st.markdown(churn_html, unsafe_allow_html= True)
  
    if CreditScore > 200 and EstimatedSalary > 10000 and IsActiveMember == 1 and NumOfProducts > 2 and Age < 50 and Tenure > 4 and Balance > 10000:
            st.success('Вероятность оттока составляет менее 20%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore > 300 and EstimatedSalary > 15000 and IsActiveMember == 1 and NumOfProducts > 3 and Age < 40 and Tenure > 5 and Balance > 20000:
            st.success('Вероятность оттока составляет менее 10%.')
            st.markdown(churn_html, unsafe_allow_html= True)

if __name__=='__main__':
    main()
