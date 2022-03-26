import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import s3fs
import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import boto3


st.title('International Football matches')


class DBConnection:
    @staticmethod
    @st.cache(ttl=600, suppress_st_warning=True)
    def get_file_from_bucket(file_path):
        dfs3 = pd.read_csv(file_path)
        return dfs3

    @staticmethod
    def fetchDB():
        uploaded_file = None
        fs = s3fs.S3FileSystem(anon=False)
        buckets = fs.ls('/')
        selected_bucket = st.selectbox('Select AWS S3 Bucket: ', buckets)

        bucket_path = 's3://'+str(selected_bucket)+'/'
        dataset_names = fs.ls(bucket_path)
        dataset = st.selectbox('Select Dataset: ', dataset_names)

        connect_btn = st.checkbox("Use selected Dataset file from Database")
        if connect_btn:
            file_path = 's3://'+str(dataset)
            #file_path = 's3://football-datasets/results.csv'
            uploaded_file = DBConnection.get_file_from_bucket(file_path)
               
        return uploaded_file




df = None

prediction_type = st.selectbox('Select Imported Dataset or Live Packet Data Prediction', ['Imported Dataset', 'Connect to AWS S3 Database'])

if prediction_type == 'Connect to AWS S3 Database':
    df = DBConnection.fetchDB()

else:
    #uploaded_file = st.file_uploader("Choose a file")
    df = pd.read_csv("assets//datasets//international-football-results//results.csv")

if df is None:
    st.stop()




st.subheader('Comparing 2 teams')
teams_to_compare = st.multiselect('Pick your teams', df['home_team'].unique())


comparison = df[(df['home_team'].isin(teams_to_compare)) & (df['away_team'].isin(teams_to_compare)) ]  
comparison = comparison.reset_index(drop=True)
st.write(comparison)
st.write('Number of matches: ', len(comparison))



#stop app if no comparison exists
if len(comparison['home_score']) == 0:
    st.stop()


st.subheader('Highest intensity of play')
out_c = comparison.iloc[np.argmax(np.array(comparison['home_score']+comparison['away_score']))]
st.write(out_c.astype(str))





team1_w = 0
team2_w = 0
teams_draw=0
team1_cum=[]
team2_cum=[]

for i in range(len(comparison)):
    if comparison['home_team'][i]==teams_to_compare[0]:
        if comparison['home_score'][i]>comparison['away_score'][i]:
            team1_w+=1
            team1_cum.append(1)
            team2_cum.append(0)
        elif comparison['home_score'][i]<comparison['away_score'][i]:
            team2_w+=1
            team1_cum.append(0)
            team2_cum.append(1)
        else:
            teams_draw+=1
            team1_cum.append(0)
            team2_cum.append(0)
    else:
        if comparison['home_score'][i]<comparison['away_score'][i]:
            team1_w+=1
            team1_cum.append(1)
            team2_cum.append(0)
        elif comparison['home_score'][i]>comparison['away_score'][i]:
            team2_w+=1
            team1_cum.append(0)
            team2_cum.append(1)
        else:
            teams_draw+=1
            team1_cum.append(0)
            team2_cum.append(0)
            
            
            
comparison_labels = ['Team 1 wins','Team 2 wins','Draws']
comparison_values = [team1_w, team2_w, teams_draw]
fig5 = go.Figure(data=[go.Pie(labels=comparison_labels, values=comparison_values)])
st.plotly_chart(fig5)





st.subheader('Cumulative wins of two teams')
 
fig6 = go.Figure()
 
fig6.add_trace(go.Scatter(x=list(df['date']), y=np.cumsum(np.array(team1_cum)), name='team 1'))
fig6.add_trace(go.Scatter(x=list(df['date']), y=np.cumsum(np.array(team2_cum)), name='team 2'))
 
 
# Add range slider
     
fig6.update_layout(
    xaxis=go.layout.XAxis(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(count=10,
                     label="10y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)
 
st.plotly_chart(fig6)




st.subheader('Frequency of city of matches')
 
cities = comparison.groupby('city').count()['country'].index.values
occurrences = comparison.groupby('city').count()['country'].values
occurrences.sort()
 
 
fig7 = go.Figure(go.Bar(
            x=occurrences,
            y=cities,
            orientation='h'))
 
 
st.plotly_chart(fig7)





st.subheader('Tournament information')
if len(comparison['home_score']) == 0:
    st.text("non monsieur")
comparison['challenge']=np.array(comparison['home_score']+comparison['away_score'])
fig8 = px.scatter(comparison, x="home_score", y="away_score",
             size="challenge", color="tournament",
                 hover_name="home_team")
 
st.plotly_chart(fig8) 




tour = st.selectbox('Select a tournament', comparison['tournament'].unique())
 
comparison_t = comparison[comparison['tournament']==tour] 
per = len(comparison_t)/len(comparison)
 
st.write(f"{round(per*100,2)}% of matches between the 2 teams have been played as {tour} matches")








sagemaker_checkbox = st.checkbox("check to use ML")
sagemaker_session = sagemaker.Session()


# Get a SageMaker-compatible role used by this Notebook Instance.
#role = 'arn:aws:iam::321439037324:role/service-role/AmazonSageMaker-ExecutionRole-20220325T184245'
role = sagemaker.get_execution_role()
#prefix = "Scikit-iris"


if sagemaker_checkbox:
    st.text("pressed")
else:
    st.text("not pressed")

os.makedirs("./data", exist_ok=True)


#connection = sqlite3.connect('data/database.sqlite')


soccer_data = pd.read_csv('players_20.csv')
soccer_data = soccer_data.fillna(soccer_data.mean())
train_input = pd.read_csv('players_20.csv')

st.table(soccer_data.head())

EPL_list = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brighton & Hove Albion', 
            'Burnley', 'Chelsea', 'Crystal Palace','Everton', 'Leicester City', 
            'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle United', 
            'Norwich City', 'Sheffield United', 'Southampton', 'Tottenham Hotspur', 
            'Watford', 'West Ham United', 'Wolverhampton Wanderers']

soccer_data['new'] = soccer_data['club'].apply(lambda x: 1 if x in EPL_list else 0)
EPL_data = soccer_data[soccer_data['new'] == 1]

EPL_data = EPL_data.dropna(axis='columns') # remove NA's
EPL_data = EPL_data[EPL_data['player_positions'] != 'GK'] # remove Goalkeepers
EPL_data = EPL_data.loc[:,~EPL_data.columns.str.contains('^goalkeeping', case=False)] # remove Goalkeepers skills 
EPL_data = EPL_data._get_numeric_data() # remove non-numerical data
EPL_data= EPL_data.drop(columns=['sofifa_id', 'new', 'value_eur', 'team_jersey_number','contract_valid_until', 'overall', 'potential'])

train_input = EPL_data
st.write("train_input file:")
st.table(train_input.head())

labels = np.log(EPL_data['wage_eur'])
st.write("labelzzzz")
st.table(labels.head())
features= EPL_data.drop('wage_eur', axis = 1)
st.write("featurezzzzz")
st.table(features.head())
feature_list = list(features.columns)
features = np.array(features)


train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
#pd.DataFrame(train_features).to_csv("data/train_features.csv")
#pd.DataFrame(train_labels).to_csv("data/train_labels.csv")
#pd.DataFrame(test_features).to_csv("data/test_features.csv")
#pd.DataFrame(test_labels).to_csv("data/test_labels.csv")


from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators = 1000, random_state = 0)
clf.fit(train_features, train_labels)

y_pred = clf.predict(test_features)
errors = abs(y_pred - test_labels)
# Print out the mean absolute error (mae)
st.write('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
fig_salaries = plt.plot(y_pred, test_labels, 'o', color='black')

fig_salaries = px.scatter(x= y_pred, y = test_labels)
st.plotly_chart(fig_salaries)







######################################## SAGEMAKER STARTS HERE ########################################################

if not sagemaker_checkbox:
    st.stop()


sess = sagemaker.Session(boto3.session.Session())
bucket = "football-analytics-bucket-sagemaker"


train_df, test_df = train_test_split(train_input, test_size = 0.25, random_state = 42)
pd.DataFrame(train_df).to_csv("data/train.csv")
#pd.DataFrame(test_df).to_csv("data/validation.csv")


FRAMEWORK_VERSION = "0.23-1"
script_path = "soccer_entry_point.py"

WORK_DIRECTORY = "data/train.csv"

train_input = sagemaker_session.upload_data(
    WORK_DIRECTORY, key_prefix="{}/{}".format("train2", WORK_DIRECTORY)
)

print(train_input)

sklearn = SKLearn(
    entry_point=script_path,
    framework_version=FRAMEWORK_VERSION,
    instance_type="ml.m5.xlarge",
    role=role,
    sagemaker_session=sagemaker_session,
)
sklearn.fit({"train": train_input})

predictor = sklearn.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge")



y_pred = clf.predict(test_features)
errors = abs(y_pred - test_labels)
# Print out the mean absolute error (mae)
st.write('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
fig_salaries = plt.plot(y_pred, test_labels, 'o', color='black')

fig_salaries = px.scatter(x= y_pred, y = test_labels)
st.plotly_chart(fig_salaries)



predictor.delete_endpoint()
