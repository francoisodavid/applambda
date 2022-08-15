import streamlit as st
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from joblib import load#,dump


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
    
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(df1.sample(10))

df2=df1.drop(['TARGET'],axis=1, inplace=False)

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

#Feature1="EXT_SOURCE_2"
#Feature2="EXT_SOURCE_3"
# Build figure
fig2=make_subplots(rows=1, cols=3, subplot_titles=("2D chart F1 vs F2", Feature1, Feature2))
#fig2 = go.Figure()
#fig2.make_subplots(rows=1, cols=2)

fig2.add_trace(
    go.Scatter(
        mode='markers',
        x=df1[Feature1],
        y=df1[Feature2],
        marker=dict(
        color='LightSkyBlue',
        size=2,
        line=dict(
            color='MediumPurple',
            width=0
            )
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


print('idc=',idc)
# Add trace with large marker
fig2.add_trace(
    go.Scatter(
        mode='markers',
        x=df1.loc[df1.index==idc][Feature1],
        y=df1.loc[df1.index==idc][Feature2],
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
x0 = df1.loc[df1.TARGET==1][Feature1]
x1 = df1.loc[df1.TARGET==0][Feature1]
fig2.add_trace(go.Histogram(x=x0,histnorm='percent', nbinsx=100,legendgroup = '2'),row=1,col=2)
fig2.add_trace(go.Histogram(x=x1,histnorm='percent', nbinsx=100,legendgroup = '2'),row=1,col=2)
fig2.add_shape(
    go.layout.Shape(type='line', xref='x', yref='paper',
                x0=df1.loc[df1.index==idc][Feature1].values[0], y0=0, x1=df1.loc[df1.index==idc][Feature1].values[0], line=dict(color="black", width=3)),
                row=1, col=2)
fig2.layout.shapes[0]['yref']='paper'
# Overlay both histograms
fig2.update_layout(barmode='overlay',title_text="Positionnement / Facteurs")
fig2.update_traces(opacity=0.75)

x1 = df1.loc[df1.TARGET==0][Feature2]
x0 = df1.loc[df1.TARGET==1][Feature2]
fig2.add_trace(go.Histogram(x=x0,histnorm='percent', nbinsx=100, legendgroup = '3'),row=1,col=3)
fig2.add_trace(go.Histogram(x=x1,histnorm='percent', nbinsx=100,legendgroup = '3'),row=1,col=3)
fig2.add_shape(
    go.layout.Shape(type='line', xref='x', yref='paper',
                x0=df1.loc[df1.index==idc][Feature2].values[0], y0=0, x1=df1.loc[df1.index==idc][Feature2].values[0], line=dict(color="black", width=3)),
                row=1, col=3)
fig2.layout.shapes[0]['yref']='paper'
# Overlay both histograms
fig2.update_layout(height=400, width=700, xaxis1_title = Feature1, yaxis1_title = Feature2,  xaxis2_title = Feature1, xaxis3_title = Feature2,  
                   barmode='overlay',title_text="Positionnement client", legend_tracegroupgap = 20,)
fig2.update_traces(opacity=0.75)



st.plotly_chart(fig2)