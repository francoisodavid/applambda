import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from joblib import load#,dump
#import shap  
import requests
from PIL import Image
st.set_page_config(layout="wide")
st.title('')
st.markdown("<h1 style='text-align: center; color: grey;'>Prédiction Risque Credit</h1>", unsafe_allow_html=True)

#@st.cache
def load_data(): #(nrows):
    data = pd.read_csv('df_for_prod.csv') #, nrows=nrows)
    dataxl = pd.read_csv('dfXL_for_prod.csv') #, nrows=nrows)
    return data,dataxl

# Create a text element and let the reader know the data is loading.
#/data_load_state = st.text('Loading data...')
df1,df1xl = load_data()
print('df1',df1.shape,df1.columns)
print('df1xl',df1xl.shape,df1xl.columns)

#df1 = pd.read_csv('df_for_prod.csv')
df2 = df1.drop(['TARGET','SK_ID_CURR','PRED','PREDproba','cluster'],axis=1, inplace=False)
df2xl = df1xl.drop(['TARGET','SK_ID_CURR','PRED','PREDproba','cluster'],axis=1, inplace=False)

print('df2',df2.shape,df2.columns)
print('df2xl',df2xl.shape,df2xl.columns)

# #data_load_state.text("Done! (using st.cache)")
# df2=df1.copy()
# df2xl=df1xl.copy()
# #print(df2)

#☺with open(f'model/model_lgb_clf_light.sav', 'rb') as f:
#    model = load('model/model_lgb_clf_light.sav')

  
if st.checkbox('Show sample data'):
    st.subheader('Raw data')
    st.write(df2.sample(5))

#df0=df2.drop(['PRED','PREDproba','cluster','TARGET','SK_ID_CURR'],axis=1, inplace=False)
#df0xl=df2xl.drop(['PRED','PREDproba','cluster','TARGET','SK_ID_CURR'],axis=1, inplace=False)

with st.sidebar:
    idc = st.selectbox('IDClient:',df1xl["SK_ID_CURR"].values)
    print(idc)
    print(df2xl.loc[df1xl["SK_ID_CURR"]==idc].shape)
    print(df2xl.columns)
    #predictions = np.round(model.predict_proba(df0.loc[df2["SK_ID_CURR"]==idc].values)[0][0],decimals=2)
    #resultat=requests.post(url='http://127.0.0.1:5000/predict',data={'SK_ID_CURR':idc}).json()
    #print('resultat=',resultat)
    predictions=requests.post(url='http://127.0.0.1:5000/predict',data={'SK_ID_CURR':idc}).json()
    print('predictions=',predictions)
    
    print('prediction=',predictions)
    st.subheader('Index de risque - Prediction')
    st.write(predictions)
    
    Age=np.round(df2xl.loc[df1xl["SK_ID_CURR"]==idc,'DAYS_BIRTH'].values[0]/-365,decimals=0)
    st.write("Age:",Age,' ans')
    
    Sexe=np.round(df2xl.loc[df1xl["SK_ID_CURR"]==idc,'CODE_GENDER'].values[0]/-365,decimals=0)
    if Sexe==1:
        Sexe="Homme"
    if Sexe==0:
        Sexe="Femme"
    st.write("Genre:  ",Sexe)
    
    Feature1list = df2xl.columns
    Feature1 = st.selectbox(
        'Facteur 1:',
        Feature1list)

    Feature2list=df2xl.columns[1:]
    Feature2 = st.selectbox(
        'Facteur 2:',
        Feature2list)
    
# si on tick on filtre les données du meme cluster que le client
print(df1xl.loc[df1xl["SK_ID_CURR"]==idc,"cluster"].values[0])
if st.checkbox('Filtre par groupes clients'):
    clu=df1xl.loc[df1xl["SK_ID_CURR"]==idc,"cluster"].values[0]
    df1xl=df1xl.loc[df1xl.cluster==clu] #je choisis une valeur par défaut
    df3=df2.copy()
    df3xl=df1xl.copy()
    st.write(df3xl.sample(2))

# on calcule une fois seulement la proba de risque de defaut
@st.cache 
def risk_proba():
    predictionsxl=np.empty((df1xl["SK_ID_CURR"].shape[0],1))
    predictionsxl[:] = np.nan    
    #print("essai:",predictionsxl.shape)
    #print('df1values',df1xl["SK_ID_CURR"].values)
    for i,idclient in enumerate(df1xl["SK_ID_CURR"].values):
        #print("i=",i,idclient)
        predictionsxli=requests.post(url='http://127.0.0.1:5000/predict',data={'SK_ID_CURR':idclient}).json()  
        predictionsxl[i]=predictionsxli['prediction']
        #mute=predictionsxl[i][0]
        #print('predictionsxli=',predictionsxl[i][0])
        #print('predictionsxli=',predictionsxli['prediction'])#["predictions"])
    return predictionsxl
#     #     ppredictionsxl = model.predict_proba(df2xl.drop(['PRED','PREDproba','cluster','TARGET','SK_ID_CURR'],axis=1, inplace=False))
#     #     return ppredictionsxl
#     idc = st.selectbox('IDClient:',df1["SK_ID_CURR"].values)
#     print(idc)
#     print(df2.loc[df1["SK_ID_CURR"]==idc].shape)
#     print(df2.columns)
#     #predictions = np.round(model.predict_proba(df0.loc[df2["SK_ID_CURR"]==idc].values)[0][0],decimals=2)
#print('toto')
predictionsxl=risk_proba()
# print('shape',ppredictionsxl.shape)
#print("tre",predictionsxl)
df3=df2.copy()
df3xl=df2xl.copy()
# #print(ppredictions[0:5,0])
print('df3xl',df3xl.shape,predictionsxl[:,0].shape)
# #print('df3',df3.shape,ppredictions[:,0].shape)
df3xl["proba"]=predictionsxl[:,0]
# #df3["proba"]=ppredictions[:,0]

if st.checkbox('Données client'):
    st.subheader('indices principaux')
    st.write(df1xl.loc[df1xl["SK_ID_CURR"]==idc])

col1, col2, col3 = st.columns(3)
with col1:
    x0=df3xl.loc[df3xl.proba>0.5,"proba"]
    x1=df3xl.loc[df3xl.proba<0.5,"proba"] # ici on a beaucoup de cas avec proba>0.5 du fait de la selection par classe 
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=x0,name="lo risk"))
    fig1.add_trace(go.Histogram(x=x1,name="hi risk"))
    print(predictions)
    fig1.add_vline(x=predictions["prediction"], line_dash = 'dash', line_color = 'firebrick')
    fig1.update_layout(height=300, width=350, title="Index Risk Client/Clientèle")#◘xaxis1_title = Feature1, yaxis1_title = Feature2)
    
    # Overlay both histograms
    # Reduce opacity to see both histograms
    fig1.update_traces(opacity=0.75)
    st.plotly_chart(fig1)
    
with col2: 
    #x0=df3[Feature1] 
    x0=df3xl.loc[df3xl.proba>0.5,Feature1]
    x1=df3xl.loc[df3xl.proba<0.5,Feature1] # inhomogene a cause de l'absence de selection sur ce facteur.
 
    x_idc2=df3.loc[df1xl["SK_ID_CURR"]==idc][Feature1] 
    print(x_idc2.values)
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=x0,name="lo risk"))
    fig2.add_trace(go.Histogram(x=x1,name="hi risk"))

    print(x_idc2.values[0]) 
    if (np.abs(x_idc2.values)>0):
        print('haha')
        fig2.add_vline(x=x_idc2.values[0], line_dash = 'dash', line_color = 'firebrick')
    fig2.update_layout(height=300, width=350, title=Feature1, barmode='stack')
    fig2.update_traces(opacity=0.75)
    st.plotly_chart(fig2)
    
with col3:
    x0=df3xl.loc[df3xl.proba>0.5,Feature2]
    x1=df3xl.loc[df3xl.proba<0.5,Feature2]
    x_idc3=df3.loc[df1xl["SK_ID_CURR"]==idc][Feature2]     
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=x0,name="lo risk"))
    fig3.add_trace(go.Histogram(x=x1,name="hi risk"))

    #print(x_idc3[idc]) 
    if (np.abs(x_idc3.values)>0):
        print('haha')
        fig3.add_vline(x=x_idc3.values[0], line_dash = 'dash', line_color = 'firebrick')
    fig3.update_layout(height=300, width=350, title=Feature2, barmode='stack')
    fig3.update_traces(opacity=0.75)
    st.plotly_chart(fig3)

col1, col2= st.columns(2)
with col1:
    st.subheader('Facteurs "clientèle"')
with col2:
    st.subheader('Facteurs "client"')

col1, col2= st.columns(2)
with col1:
    image = Image.open('static/images/imageshap.png')
    st.image(image, caption='Importance of factors globally', width=400)
# with col2:
#     image = Image.open('static/images/lime_'+str(idc)+'.png')
#     st.image(image, caption='Importance of factors in that case', width=400)

# Build figure
fig10=make_subplots(rows=1, cols=1, subplot_titles=("Factor interaction", Feature1, Feature2))
#fig2 = go.Figure()
#fig2.make_subplots(rows=1, cols=2)

fig10.add_trace(
    go.Scatter(
        mode='markers',
        x=df3xl[Feature1],
        y=df3xl[Feature2],
        name="low risk",
        marker=dict(
        color='LightSkyBlue',
        size=2,
        line=dict(
            color='MediumPurple',
            width=0
            ),
        
        ),legendgroup = '1',
        showlegend=True
        ),
    row=1,col=1
    )
fig10.add_trace(
    go.Scatter(
        mode='markers',
        x=df3xl.loc[df1xl.TARGET==1][Feature1],
        y=df3xl.loc[df1xl.TARGET==1][Feature2],        
        name="high risk",
        marker=dict(
        color='crimson',
        size=0,
        line=dict(
            color='MediumPurple',
            width=1
            )
        ),legendgroup = '1',
        showlegend=True
        ),row=1,col=1
    )    

# Add trace with large marker
fig10.add_trace(
    go.Scatter(
        mode='markers',
        x=df3xl.loc[df1xl["SK_ID_CURR"]==idc][Feature1],
        y=df3xl.loc[df1xl["SK_ID_CURR"]==idc][Feature2],        
        name="Client risk",
        marker=dict(
            color='black',
            size=30,
            line=dict(
                color='MediumPurple',
                width=3
            )
        ),legendgroup = '1',
        showlegend=True
    ),row=1,col=1            
)    
fig10.update_layout(height=800, width=800, xaxis1_title = Feature1, yaxis1_title = Feature2)
#                    barmode='overlay',title_text="Positionnement client", legend_tracegroupgap = 20,)
# fig2.update_traces(opacity=0.75)
st.write('')
st.write('')
st.subheader('Données individuelles')
st.plotly_chart(fig10)

