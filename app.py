import sklearn

#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
#from lightgbm import LGBMClassifier
import xgboost as xgb

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


classifier_name=['LightGMBClassifier']
option = st.sidebar.selectbox('Выберите алгоритм для прогнозирования', classifier_name)
st.subheader(option)



#Importing model and label encoders
model = pickle.load(open("final_LGBM_model.pkl","rb"))
#model_1 = pickle.load(open("final_RF_model.pkl","rb"))
#model_2 = pickle.load(open("final_LR_model.pkl","rb"))
le_pik=pickle.load(open("label_encoding_for_gender.pkl","rb"))
le1_pik=pickle.load(open("label_encoding_for_geo.pkl","rb"))

def predict_churn(CreditScore, Geo, Gen, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):
    input = np.array([[CreditScore, Geo, Gen, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]).astype(np.float64)
    if option == 'LightGMBClassifier':
        prediction = model.predict_proba(input)
        pred = '{0:.{1}f}'.format(prediction[0][0], 2)

    #if option == 'RandomForestClassifier':
       #prediction = model_1.predict_proba(input)
        #pred = '{0:.{1}f}'.format(prediction[0][0], 2)
        
    #if option == 'LogisticRegression':
        #prediction = model_2.predict_proba(input)
        #pred = '{0:.{1}f}'.format(prediction[0][0], 2)

    #else:
        #pred=0.30
        #st.markdown('Клиент остаётся в банке')

    return float(pred)


def main():
    st.title("Прогноз оттока клиентов из банка")
    html_temp = """
    <div style="background-color:#f3f6f4 ;padding:10px">
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
        
    Age = st.slider('Возраст', 18, 95)

    Tenure = st.slider('Срок обслуживания в банке, лет', 0, 10)

    Balance = st.slider('Баланс счёта',  0, 5000)

    NumOfProducts = st.slider('Количество банковских продуктов', 0, 4)

    HasCrCard = st.selectbox('Наличие кредитной карточки', ['No', 'Yes'])
    
    if HasCrCard == 'No':
        HasCrCard = 0
    else:
        HasCrCard = 1
    
    IsActive = st.selectbox("Активность клиента", ['No', 'Yes'])
    
    if IsActive == 'No':
        IsActiveMember = 0
    else:
        IsActiveMember = 1

    EstimatedSalary = st.slider('Предполагаемая заработная плата',  0, 5000)

    
    if CreditScore < 100 and Balance < 500 and EstimatedSalary < 500 and IsActiveMember == 0 and NumOfProducts == 0 and Tenure == 0:
            st.success('Вероятность оттока составляет более 90%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore < 200 and Balance < 1000 and EstimatedSalary < 1000 and IsActiveMember == 0 and NumOfProducts == 0 and Tenure == 1:
            st.success('Вероятность оттока составляет более 70%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore < 300 and Balance < 1500 and EstimatedSalary < 1500 and IsActiveMember == 0 and NumOfProducts == 0 and Tenure == 2:
            st.success('Вероятность оттока составляет более 50%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore > 100 and EstimatedSalary > 500 and IsActiveMember == 1 and Age < 60 and Tenure > 3 and Balance > 500:
            st.success('Вероятность оттока составляет менее 30%.')
            st.markdown(churn_html, unsafe_allow_html= True)
  

                   
    churn_html = """  
              <div style="background-color:#f44336;padding:20px >
               <h2 style="color:red;text-align:center;">👎 К сожалению, мы теряем клиента...</h2>
               </div>
            """
    no_churn_html = """  
              <div style="background-color:#94be8d;padding:20px >
               <h2 style="color:green ;text-align:center;">👌 Успех, клиент остаётся в банке!</h2>
               </div>
            """
            
            
    if CreditScore < 100 and Balance < 500 and EstimatedSalary < 500 and IsActiveMember == 0 and NumOfProducts == 0 and Tenure == 0:
            st.success('Вероятность оттока составляет более 90%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore < 200 and Balance < 1000 and EstimatedSalary < 1000 and IsActiveMember == 0 and NumOfProducts == 0 and Tenure == 1:
            st.success('Вероятность оттока составляет более 70%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore < 300 and Balance < 1500 and EstimatedSalary < 1500 and IsActiveMember == 0 and NumOfProducts == 0 and Tenure == 2:
            st.success('Вероятность оттока составляет более 50%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore > 100 and EstimatedSalary > 500 and IsActiveMember == 1 and Age < 60 and Tenure > 3 and Balance > 500:
            st.success('Вероятность оттока составляет менее 30%.')
            st.markdown(churn_html, unsafe_allow_html= True)
  
    if CreditScore > 200 and EstimatedSalary > 1000 and IsActiveMember == 1 and Age < 50 and Tenure > 4 and Balance > 1000:
            st.success('Вероятность оттока составляет менее 20%.')
            st.markdown(churn_html, unsafe_allow_html= True)

    if CreditScore > 300 and EstimatedSalary > 1500 and IsActiveMember == 1 and  Age < 40 and Tenure > 5 and Balance > 2000:
            st.success('Вероятность оттока составляет менее 10%.')
            st.markdown(churn_html, unsafe_allow_html= True)
    if st.button('Сделать прогноз'):
        output = predict_churn(CreditScore, Geo, Gen, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary)
        st.success('Вероятность оттока составляет {}'.format(output))
        

        if output >= 0.5:
            st.markdown(churn_html, unsafe_allow_html= True)

        else:
            st.markdown(no_churn_html, unsafe_allow_html= True)
            
            
        
if __name__=='__main__':
    main()
