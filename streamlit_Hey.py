import streamlit as st
import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
import string
from nltk.corpus import stopwords
import nltk


# page config
st.set_page_config(
    page_title="Analítica de HeyBot!",
    page_icon="✖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the DataFrame
df = pd.read_csv("E:\data\Extended Dataset hey.csv")
metrics = ['Comments','Likes', 'Retweets', 'Analytics']

# nav bar
with st.sidebar:
    st.title('Analítica de HeyBot!')
    labeldict = {'Omisión': 0, 'Atención': 1, 'Oportunidad': 2}
    label = st.selectbox('Clase', ['Omisión', 'Atención', 'Oportunidad'])
    metric = st.selectbox('Métrica', metrics)

labeldict_rev = {0: 'Omisión', 1: 'Atención', 2: 'Oportunidad'}

# Filter data for each label
label = labeldict[label]
labeldf = df[df['label'] == label]

# Preprocess the 'Tags' column
df['Tags'] = df['Tags'].apply(lambda x: str(x).replace('[', '').replace(']', '').replace("'", '').replace(",", ''))

#time series
for i in range(len(labeldf['Analytics'])):
    labeldf['Analytics'].iloc[i] = re.sub(r'\s[Mmil]*', '000', labeldf['Analytics'].iloc[i])
labeldf['Analytics'] = labeldf['Analytics'].astype(int)

for i in range(len(labeldf['Likes'])):
    labeldf['Likes'].iloc[i] = re.sub(r'\s[Mmil]*', '000', labeldf['Likes'].iloc[i])
labeldf['Likes'] = labeldf['Likes'].astype(int)

fig=px.line(x=labeldf[labeldf['label']==label]['Date'].sort_values(),y=labeldf[labeldf['label']==label][metric])
fig.update_layout(yaxis=dict(range=[0, labeldf[labeldf['label']==label][metric].max()*1.1]))
fig.update_xaxes(title_text='Fecha')
fig.update_yaxes(title_text='Suma')

#pie chart
nltk.download('stopwords')
piedf = df[df['label'] == 2]
text = ' '.join(piedf['Content'])
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))
words = text.split()
stop_words = set(stopwords.words('spanish')) 
filtered_words = [word for word in words if word.lower() not in stop_words]
word_freq = Counter(words)
dictionary = dict(word_freq)
filtered_dict = {key: value for key, value in dictionary.items() if key in ['mercado', 'santander', 'bbva', 'nu', 'neobanco', 'story', 'banregio', 'azteca', 'bancoazteca', 'platacard', 'bitso', 'binanace', 'mercadopago', 'banamex', 'heybanco', 'nubank']}
filtered_dict['bancoazteca'] = filtered_dict['azteca'] + filtered_dict['bancoazteca']
del filtered_dict['azteca']
filtered_dict['nubank'] = filtered_dict['nu'] + filtered_dict['nubank']
del filtered_dict['nu']
df__ = pd.DataFrame(list(filtered_dict.items()), columns=['Banco', 'Frecuencia'])
fig_pie = px.pie(df__, values='Frecuencia', names='Banco')

# Function to generate word cloud and save as image
width = 800
height = 400

def generate_wordcloud_and_save(tags_string, filename):
    wordcloud = WordCloud(width=width, height=height, background_color='white').generate(tags_string)
    wordcloud.to_file(filename)


# Generate word clouds and save as images
generate_wordcloud_and_save(re.sub(r'\s+', ' ', labeldf['Tags'].str.cat(sep=' ')), "wordcloud1.png")


col = st.columns(2, gap='small')

# Tags for cases of Attention
with col[0]:
    col_ = st.columns(2, gap='small')
    for i, column in enumerate(metrics):
        if i < 2:
            with col_[0]:
                st.metric(column, labeldf[column].sum())
                if i == 1:
                    st.metric('Clasificados en clase ' + labeldict_rev[label], labeldf.shape[0])
                    st.metric('Proporción', f'%i %%' % (labeldf.shape[0] / df.shape[0] * 100))
        else: 
            with col_[1]:
                st.metric(column, labeldf[column].sum())
                if i == 3:
                    st.metric('Registros', df.shape[0])
    
with col[0]:
    st.markdown('## Aparición de otros bancos')
    st.write(fig_pie)
    
   
# Tags for cases of Opportunity
with col[1]:
    st.markdown('## Análisis temporal')
    st.write(fig)
    st.markdown('## Nube de palabras')
    st.image("wordcloud1.png", width=500)

st.markdown('## Análisis Tabular')
table_height = 300  
st.table(labeldf.style.set_table_attributes(f'style="height: {table_height}px;"'))
