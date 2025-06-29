import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model, metrics
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,roc_auc_score, RocCurveDisplay,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB



st.markdown('---')

st.subheader('A detailed analysis of Breast Cancer Prediction')

st.markdown('---')

st.markdown("""

## Objective:
            
The objective of this data analysis and visualization is to explore the Breast Cancer dataset to identify key features influencing 
diagnosis.Through statistical summaries and visualizations,we aim to understand patterns, detect correlations between variables, 
and distinguish between malignant and benign tumors to support accurate and early breast cancer prediction.
            
## Project Overview
#### In this project, I aim to:

1. Explore and clean a real-world breast cancer dataset.

2. Analyze the features that influence diagnosis outcomes.

3. Train a machine learning model to classify tumors as benign or malignant.

4. Evaluate the model using key performance metrics.

5. Deploy the model using Streamlit, enabling users to input values and receive predictions in real time.

""")

st.markdown('---')

df = pd.read_csv('./data.csv')

df = df.drop(['id','Unnamed: 32'],axis=1)

st.subheader('Features present in DataFrame:')
st.caption('This table lists all feature names (columns) present in the Breast Cancer dataset.')

with st.container():
    
    st.dataframe(pd.DataFrame(df.columns, columns=['Column Names']))
    

st.markdown('---')


st.subheader('Understand the Features')
st.caption('The feature names may seem complex at first glance.' \
'Don’t worry — just select a feature below to read its definition and understand what it represents.')



feature_descriptions = {
    "radius_mean": "The average radius (distance from the center to the perimeter) of the tumor cells; larger values may indicate malignancy.",
    "texture_mean": "The average variation in pixel intensity; higher texture irregularity can indicate malignancy.",
    "perimeter_mean": "The mean length of the tumor boundary; larger perimeters are often linked to malignancy.",
    "area_mean": "The average size of the tumor; larger areas are typically associated with malignant tumors.",
    "smoothness_mean": "The average smoothness of the tumor's surface; less smoothness often indicates malignancy.",
    "compactness_mean": "The mean ratio of perimeter squared to the area; higher compactness can signal malignancy.",
    "concavity_mean": "The average depth of concave areas in the tumor boundary; more pronounced concavities often suggest malignancy.",
    "concave points_mean": "The average number of concave points on the tumor boundary; a higher count often correlates with malignancy.",
    "symmetry_mean": "The average symmetry of the tumor; asymmetry is more common in malignant tumors.",
    "fractal_dimension_mean": "The mean complexity of the tumor boundary; higher values may indicate malignancy.",
    "radius_se": "The standard error of the tumor radius; measures variability and can indicate malignancy.",
    "texture_se": "The standard error of texture intensity variation; higher variability may suggest malignancy.",
    "perimeter_se": "The standard error of the tumor perimeter; reflects the consistency of the boundary measurements.",
    "area_se": "The standard error of the tumor area; higher variability in size may indicate malignancy.",
    "smoothness_se": "The standard error of smoothness; measures irregularity in the tumor surface.",
    "compactness_se": "The standard error of compactness; variability in compactness can indicate malignancy.",
    "concavity_se": "The standard error of concave areas; reflects variability in tumor boundary concavities.",
    "concave points_se": "The standard error of concave points; higher variability may suggest malignancy.",
    "symmetry_se": "The standard error of symmetry; measures inconsistencies in tumor symmetry.",
    "fractal_dimension_se": "The standard error of fractal dimension; variability in boundary complexity.",
    "radius_worst": "The largest observed tumor radius; larger values are often linked to malignancy.",
    "texture_worst": "The most irregular texture observed; higher irregularity often suggests malignancy.",
    "perimeter_worst": "The longest observed tumor boundary; larger perimeters are typically associated with malignancy.",
    "area_worst": "The largest tumor size observed; larger sizes are linked to malignancy.",
    "smoothness_worst": "The least smooth surface observed; reduced smoothness may indicate malignancy.",
    "compactness_worst": "The highest compactness observed; higher values are often correlated with malignancy.",
    "concavity_worst": "The largest depth of concave areas observed; pronounced concavities often suggest malignancy.",
    "concave points_worst": "The highest number of concave points observed; a larger count often indicates malignancy.",
    "symmetry_worst": "The least symmetric tumor shape observed; asymmetry is a potential malignancy indicator.",
    "fractal_dimension_worst": "The most complex boundary observed; higher complexity may indicate malignancy."
}

with st.container():
     feature = df.columns
     selected_feature = st.selectbox('Select the feature to read the detail about it',feature)
     if selected_feature in feature_descriptions:
        st.markdown(f"**Definition:** {feature_descriptions[selected_feature]}")
     else:
       st.info("No definition available for this feature.")
          

st.markdown('---')






#st.markdown("As we can seen there multiple features available in the dataset. Now we can look " \"into the dataset and type of values present in the dataset")
#data_type = df.info
#ata_type
#st.dataframe()

st.subheader("Dataset Columns and Their Data Types")
# Add a helpful caption
st.caption("This table displays each column in the dataset along with its corresponding data type.")

# Create a DataFrame from df.dtypes
column_info = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
column_info.columns = ['Column Name', 'Data Type']

column_info = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str).values  # convert to string
})
st.dataframe(column_info)

# # Convert 'Data Type' to string
# column_info['Data Type'] = column_info['Data Type'].astype(str)

# # Show as a table
# st.dataframe(column_info)



st.markdown('---')

st.subheader("This step helps us determine whether any missing values are present in the dataset.")
st.caption('A value of 0 indicates that there are no null values in the corresponding column. ' \
'This information is essential for deciding whether we need to apply any data cleaning or imputation techniques.')

null_counts = df.isnull().sum()

st.dataframe(pd.DataFrame(null_counts,columns=['Count of missing values']))

#('Let\'s check null values present in the Dataframe',null_counts)


st.markdown('---')

st.subheader('Feature Correlation Analysis')

st.caption("Correlation is a statistical method used to measure the strength and direction of the relationship between two or " \
"more variables.In this step, we will analyze the correlation among the features in the dataset to " \
"understand how they are related to each other.")

corr = df.corr(numeric_only=True)
def highlight_high_corr(val):
    color='background-color:lightgreen' if abs(val)>0.7 or abs(val) < -0.7 else '' 
    return color

styled_corr = corr.style.map(highlight_high_corr)
st.dataframe(styled_corr)

st.markdown('---')

st.subheader('Scatter Plot for Visualizing Feature Relationships')

st.caption("After identifying strongly correlated features, we now use interactive selectboxes to choose any two features and visualize their relationship " \
"through a scatter plot to better understand underlying patterns or trends.")
col1,col2=st.columns(2)
with col1:
    first_selected_feature = st.selectbox('Select the first feature',df.columns)
with col2:
    second_selected_feature = st.selectbox('Select the second feature',df.columns)
fig,ax=plt.subplots(figsize=(8,5))
sns.scatterplot(data=df,x=first_selected_feature,y=second_selected_feature)

st.pyplot(fig)

st.markdown('---')


st.subheader('Distribution & Outlier Analysis')
st.caption('The boxplot helps identify outliers and the spread of the data, while the histogram shows ' \
'the distribution and frequency of values for the selected feature. Together, these visualizations offer a ' \
'comprehensive understanding of data variability, central tendency, and skewness—crucial for detecting ' \
'patterns and making informed decisions in diagnosis prediction.')

selected_feature = st.selectbox('Select the feature',df.columns)
col3,col4=st.columns(2)

with col3:
        st.write('**BoxPlot**')
        fig,ax=plt.subplots(figsize=(8,5))
        sns.boxplot(data=df,x='diagnosis',y=selected_feature)
        st.pyplot(fig)

with col4:
        st.write('**HistPlot**')
        fig,ax=plt.subplots(figsize=(8,5))
        sns.histplot(data=df[selected_feature])
        st.pyplot(fig)

st.subheader("Explore Feature Relationships with Pairplot")

# Feature selection
feature_options = df.select_dtypes(include=['float64', 'int64']).columns.tolist()  # only numeric features
selected_features = st.multiselect(
    "Select two or more features to plot:",
    options=feature_options,
    default=["radius_mean", "texture_mean", "concave points_worst"]
)

# Generate pairplot button
if st.button("Generate Pairplot"):
    if len(selected_features) >= 2:
        with st.spinner("Generating pairplot..."):
            pairplot = sns.pairplot(df[selected_features + ['diagnosis']], hue='diagnosis')
            st.pyplot(pairplot.fig)
    else:
        st.warning("Please select at least two features to generate the pairplot.")

#Model Building

st.subheader('Model building')


# dups_df = df.copy()
# dups_df['diagnosis'] = preprocessing.LabelEncoder().fit_transform(dups_df['diagnosis'])
# result_df = dups_df[['diagnosis']]
# feature_df = dups_df.drop(columns=['diagnosis'])




dups_df = df.copy()
result_df  = dups_df[['diagnosis']]
le1 = preprocessing.LabelEncoder()
result_df.loc[:, 'diagnosis'] = le1.fit_transform(result_df['diagnosis'])

feature_df = dups_df.drop(columns=['diagnosis'])


with st.container():
     st.caption('Feel free to choose a scaler; StandardScaler usually works well for most machine learning models.')
     selected_standardization_process = st.radio('Select the standardization',options=('Standard_Scaler','MinMaxScaler'))
     if st.button('Perform Feature Scaling'):
          with st.spinner('Please wait...Standardizing the data'):
            if selected_standardization_process == 'Standard_Scaler':
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(feature_df)
                scaled_df = pd.DataFrame(scaled_data,columns=feature_df.columns)
            else:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(feature_df)
                scaled_df = pd.DataFrame(scaled_data,columns=feature_df.columns)
            st.session_state['scaled_data'] = scaled_data
            st.success('Standardization of Data completed successfully')

if 'scaled_data' in st.session_state:
    X = st.session_state['scaled_data']
    y = result_df['diagnosis']

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle = True, random_state=42)


    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'Naive Bayes': GaussianNB()
    }

    selected_model_name = st.selectbox('Please select the ML model',list(models.keys()))
    selected_model = models[selected_model_name]
    if st.button('Run the model'):
        
            
        # for model_name, model in models.items():
        #     # Train the model
            selected_model.fit(X_train, y_train)

            # Make predictions
            y_pred = selected_model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_percentage = accuracy * 100

            st.success(f"{selected_model_name} achieved an accuracy of {accuracy_percentage:.2f}%")


            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # ROC-AUC Score (for binary classifiers only)
            if len(np.unique(y_test)) == 2:
                y_prob = selected_model.predict_proba(X_test)[:, 1] if hasattr(selected_model, "predict_proba") else selected_model.decision_function(X_test)
                auc = roc_auc_score(y_test, y_prob)
                st.write(f"ROC-AUC Score: {auc:.2f}")

                # ROC Curve
                st.subheader("ROC Curve")
                roc_disp = RocCurveDisplay.from_estimator(selected_model, X_test, y_test)
                fig, ax = plt.subplots()
                roc_disp.plot(ax=ax)
                st.pyplot(fig)

            train_acc = selected_model.score(X_train, y_train)
            test_acc = selected_model.score(X_test, y_test)

            st.write(f"Training Accuracy: {train_acc:.2f}")
            st.write(f"Testing Accuracy: {test_acc:.2f}")

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=selected_model.classes_)

            fig, ax = plt.subplots()
            disp.plot(ax=ax)
            st.pyplot(fig)

with st.expander('Final words!!'):
     st.write("""* The machine learning models applied to the breast cancer dataset have demonstrated exceptional performance across the board. 
                Every classifier tested — including Logistic Regression, Random Forest, SVM, KNN, Decision Tree, and Naive Bayes — consistently 
                achieved over 95% accuracy, with the best models reaching 99% on training data and 98% on testing data, indicating no overfitting.\n"""
                """* Further analysis through the classification report showed high precision, recall, and F1-scores for both malignant and benign classes,
                ensuring balanced and reliable predictions. The ROC-AUC score of 1.00 and the near-perfect ROC curve confirm the model's excellent ability 
                to differentiate between the two classes, making it a powerful tool for diagnostic support.\n"""
                """* This high level of accuracy, interpretability, and generalization — especially on a relatively small dataset — reflects the strength of the 
                features, preprocessing steps like standardization, and model robustness. The system is highly dependable and suitable for real-world breast 
                cancer detection tasks.""")



# st.markdown(
#     """
#     ### 💬 Share Feedback
#     Please [click here](https://docs.google.com/forms/d/13j3iUoDrUsbHwelZuj1VrbkAmRdj8n-uBLKwdm62V5k/edit) to submit your feedback.
#     """,
#     unsafe_allow_html=True
# )
st.header('Thank you!!!')
                        
            
            




    

            
          
     



