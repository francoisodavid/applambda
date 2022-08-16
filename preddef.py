import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from joblib import load#,dump
#import shap
from PIL import Image

st.title('Simple Credit Prediction')

@st.cache
def load_data(nrows):
    data = pd.read_csv('df_for_prod.csv', nrows=nrows)
    return data
# Create a text element and let the reader know the data is loading.
#/data_load_state = st.text('Loading data...')
df1 = load_data(10000)
#data_load_state.text("Done! (using st.cache)")




with open(f'model/model_lgb_clf_light.sav', 'rb') as f:
    model = load(f)
    
if st.checkbox('Show sample data'):
    st.subheader('Raw data')
    st.write(df1.sample(5))

df2=df1.drop(['PRED','cluster','TARGET'],axis=1, inplace=False)

with st.sidebar:
    idc = st.selectbox('IDClient:',df2.index)
    predictions = np.round(model.predict_proba(df2.loc[df2.index==idc].values)[0][0],decimals=2)

    st.subheader('Risk Prediction')
    st.write(predictions)
    
    Feature1list=df2.columns
    Feature1 = st.selectbox(
        'Facteur 1:',
        Feature1list)

    Feature2list=df2.columns[1:]
    Feature2 = st.selectbox(
        'Facteur 2:',
        Feature2list)

# si on tick on filtre les données du meme cluster que le client
if st.checkbox('Filter data'):
    df1=df1.loc[df1.cluster==df1['cluster'].iloc[idc]] #je choisis une valeur par défaut
    st.write(df1.sample(5))

# on calcule une fois seulement la proba de risque de defaut
@st.cache
def risk_proba():
    ppredictions = model.predict_proba(df2)
    return ppredictions
ppredictions=risk_proba()
df3=df2.copy()
#st.write(ppredictions[0:5,0])
df3["proba"]=ppredictions[:,0]


if st.checkbox('Show client data'):
    st.subheader('Raw data')
    st.write(df1.iloc[idc])

x0=df3.loc[df3.proba<0.5,"proba"]
x1=df3.loc[df3.proba>0.5,"proba"]
fig1 = go.Figure()
fig1.add_trace(go.Histogram(x=x0,name="low risk"))
fig1.add_trace(go.Histogram(x=x1,name="high risk"))
print(predictions)
fig1.add_vline(x=predictions, line_dash = 'dash', line_color = 'firebrick')
fig1.update_layout(height=400, width=600, title="Risk factor client")#◘xaxis1_title = Feature1, yaxis1_title = Feature2)

# Overlay both histograms
# Reduce opacity to see both histograms
fig1.update_traces(opacity=0.75)
st.plotly_chart(fig1)

image = Image.open('imageshap.png')

st.image(image, caption='Importance of factors wrt to risk', width=400)


# Build figure
fig2=make_subplots(rows=1, cols=1, subplot_titles=("Factor interaction", Feature1, Feature2))
#fig2 = go.Figure()
#fig2.make_subplots(rows=1, cols=2)

fig2.add_trace(
    go.Scatter(
        mode='markers',
        x=df1[Feature1],
        y=df1[Feature2],
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
fig2.add_trace(
    go.Scatter(
        mode='markers',
        x=df1.loc[df1.TARGET==1][Feature1],
        y=df1.loc[df1.TARGET==1][Feature2],        
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
fig2.add_trace(
    go.Scatter(
        mode='markers',
        x=df1.loc[df1.index==idc][Feature1],
        y=df1.loc[df1.index==idc][Feature2],        
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
fig2.update_layout(height=400, width=400, xaxis1_title = Feature1, yaxis1_title = Feature2)
#                    barmode='overlay',title_text="Positionnement client", legend_tracegroupgap = 20,)
# fig2.update_traces(opacity=0.75)



st.plotly_chart(fig2)
# x0 = df1.loc[df1.TARGET==1][Feature1]
# x1 = df1.loc[df1.TARGET==0][Feature1]
# fig2.add_trace(go.Histogram(x=x0,histnorm='percent', nbinsx=50,legendgroup = '2'),row=1,col=2)
# fig2.add_trace(go.Histogram(x=x1,histnorm='percent', nbinsx=50,legendgroup = '2'),row=1,col=2)
# fig2.add_shape(
#     go.layout.Shape(type='line', xref='x', yref='paper',
#                 x0=df1.loc[df1.index==idc][Feature1].values[0], y0=0, x1=df1.loc[df1.index==idc][Feature1].values[0], line=dict(color="black", width=3)),
#                 row=1, col=2)
# fig2.layout.shapes[0]['yref']='paper'
# # Overlay both histograms
# fig2.update_layout(barmode='overlay',title_text="Positionnement / Facteurs")
# fig2.update_traces(opacity=0.75)

# x1 = df1.loc[df1.TARGET==0][Feature2]
# x0 = df1.loc[df1.TARGET==1][Feature2]
# fig2.add_trace(go.Histogram(x=x0,histnorm='percent', nbinsx=50, legendgroup = '3'),row=1,col=3)
# fig2.add_trace(go.Histogram(x=x1,histnorm='percent', nbinsx=50,legendgroup = '3'),row=1,col=3)
# fig2.add_shape(
#     go.layout.Shape(type='line', xref='x', yref='paper',
#                 x0=df1.loc[df1.index==idc][Feature2].values[0], y0=0, x1=df1.loc[df1.index==idc][Feature2].values[0], line=dict(color="black", width=3)),
#                 row=1, col=3)
# fig2.layout.shapes[0]['yref']='paper'
# # Overlay both histograms
# fig2.update_layout(height=400, width=700, xaxis1_title = Feature1, yaxis1_title = Feature2,  xaxis2_title = Feature1, xaxis3_title = Feature2,  
#                    barmode='overlay',title_text="Positionnement client", legend_tracegroupgap = 20,)
# fig2.update_traces(opacity=0.75)



# st.plotly_chart(fig2)