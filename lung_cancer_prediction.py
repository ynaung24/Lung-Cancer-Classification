import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, roc_auc_score
from itertools import combinations

# Set wide mode for Streamlit app layout
st.set_page_config(layout="wide")

# Custom CSS to adjust spacing
st.markdown("""
    <style>
        .custom-title {
            font-size: 16px;  /* Reduced font size */
            font-weight: bold;
            margin-top: 10px;  /* Preserved margin for titles */
        }
        .stSlider {
            margin-top: -10px;  /* Adjust space above the slider */
        }
        .stCheckbox {
            margin-bottom: -10px;  /* Removed space between checkboxes */
        }
    </style>
""", unsafe_allow_html=True)

# Modify the load_data function
@st.cache_data
def load_data():
    df = pd.read_csv('cancer.csv')
    df.columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
                  'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 
                  'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 
                  'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 
                  'CHEST PAIN', 'LUNG_CANCER']
    df = df.replace({1: 0, 2: 1})

    # Encode categorical variables
    df["GENDER"] = df['GENDER'].map({'M': 1, 'F': 0})
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, "NO": 0})
    return df, 'LUNG_CANCER'

# Add the new fit_models function
@st.cache_data
def fit_models(data, target_column):
    y = data[target_column]
    X = data.drop(columns=[target_column])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)
    
    predictors = X.columns.tolist()
    
    results = []
    
    for i in range(1, len(predictors) + 1):
        for combo in combinations(predictors, i):
            X_train_subset = X_train[list(combo)]
            X_test_subset = X_test[list(combo)]
            
            model = LogisticRegression(solver='liblinear', random_state=24)
            model.fit(X_train_subset, y_train)
            y_pred = model.predict(X_test_subset)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)
            
            results.append({
                'Combination': ', '.join(combo),
                'Number of Predictors': len(combo),
                'ROC_AUC': roc_auc,
                'Accuracy': accuracy,
                'F1 Score': f1,
                'Coefficients': model.coef_[0]
            })

    results_sorted = sorted(results, key=lambda x: x['F1 Score'], reverse=True)
    
    return results_sorted

# Load data and fit models
df, target_column = load_data()
model_results = fit_models(df, target_column)

# Use the best model (highest F1 score) for predictions
best_model = model_results[0]
best_predictors = best_model['Combination'].split(', ')

# Train the best model
X_best = df[best_predictors]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X_best, y, test_size=0.2, random_state=24)
model = LogisticRegression(solver='liblinear', random_state=24)
model.fit(X_train, y_train)

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = 'Predictor-Dependent Analysis'

# Sidebar for navigation using buttons
st.sidebar.title("Navigation")
if st.sidebar.button("Predictor-Dependent Analysis", key="predictor"):
    st.session_state.page = 'Predictor-Dependent Analysis'
if st.sidebar.button("Non-Predictor Dependent Analysis", key="non_predictor"):
    st.session_state.page = 'Non-Predictor Dependent Analysis'

# Predictor-Dependent Analysis Page
if st.session_state.page == 'Predictor-Dependent Analysis':
    st.markdown('<div class="custom-title">Lung Cancer Prediction and Feature Impact</div>', unsafe_allow_html=True)

    # Set up horizontal checkboxes for feature selection
    st.markdown('<div class="custom-title">Select Features to Analyze their Impact</div>', unsafe_allow_html=True)

    # Use only the best predictors for selection, excluding AGE
    predictors = [pred for pred in best_predictors if pred != 'AGE']

    # Create a list of selected predictors using checkboxes arranged horizontally
    selected_predictors = []
    num_cols = 4  # Number of checkboxes per row
    rows = [predictors[i:i + num_cols] for i in range(0, len(predictors), num_cols)]

    for row in rows:
        cols = st.columns(num_cols)
        for col, predictor in zip(cols, row):
            if col.checkbox(predictor, key=predictor):
                selected_predictors.append(predictor)

    # Add AGE slider
    age = st.slider("AGE", min_value=0, max_value=100, value=50, step=1)

    # Display selected predictors
    if selected_predictors:
        st.write(f"Currently selected predictors: {', '.join(selected_predictors)}")
    else:
        st.write("No predictors selected. Please select predictors to analyze their impact.")
    
    # Calculate probability of lung cancer based on selected features
    input_data = {}
    for feature in best_predictors:
        if feature == 'AGE':
            input_data[feature] = age
        elif feature in selected_predictors:
            input_data[feature] = 1  # Set selected features to 1 (checked)
        else:
            input_data[feature] = 0  # Default to 0 (not selected)

    # Convert input data into DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    # Ensure input_df has the same columns as X_train
    input_df = input_df[X_train.columns]

    # Make prediction and display as a percentage
    prediction_percentage = model.predict_proba(input_df)[0][1] * 100  # Convert to percentage

    # Display probability of lung cancer based on selected predictors with larger font and red color
    st.markdown(f'<div style="font-size: 24px; font-weight: bold; color: red;">Probability of Lung Cancer: {prediction_percentage:.2f}%</div>', unsafe_allow_html=True)

    # Interactive Feature Impact Plot
    st.markdown('<div class="custom-title">Interactive Feature Impact</div>', unsafe_allow_html=True)

    fig = go.Figure()

    # Clean plotting to avoid double lines and show impact dynamically based on selected features
    for feature in selected_predictors:
        if feature in best_predictors and feature != 'AGE':  # Exclude AGE from the plot
            feature_values = np.array([0, 1])  # Binary features
            
            # Calculate impact using the correct index from best_predictors
            feature_index = best_predictors.index(feature)
            impact = model.coef_[0][feature_index] * feature_values

            fig.add_trace(go.Scatter(x=feature_values, y=impact, mode='lines', name=feature))

    if len(fig.data) > 0:  # Only update layout if there are traces to plot
        fig.update_layout(
            title="Effect of Features on Lung Cancer Probability",
            xaxis=dict(title="Feature Value", titlefont=dict(size=14)),
            yaxis=dict(title="Impact on Odds of Lung Cancer", titlefont=dict(size=14)),
            showlegend=True,
            height=350
        )
        st.plotly_chart(fig)
    else:
        st.write("No selected features are part of the best model. Please select features used in the model to see their impact.")

# Non-Predictor Dependent Analysis Page
elif st.session_state.page == 'Non-Predictor Dependent Analysis':
    st.markdown('<div class="custom-title">Non-Predictor Dependent Model Analysis</div>', unsafe_allow_html=True)

    # Hide the sidebar input features when on this page
    st.sidebar.empty()

    # Use st.columns to show the graphs side by side
    col1, col2 = st.columns(2)

    with col1:
        # Feature Importance with smaller height
        st.markdown('<div class="custom-title">Feature Importance</div>', unsafe_allow_html=True)
        coef_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': model.coef_[0]})
        coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
        fig_imp = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h', height=350)  # Reduced height
        fig_imp.update_layout(
            xaxis=dict(title="Coefficient", titlefont=dict(size=14)),  # Reduced font size
            yaxis=dict(title="Feature", titlefont=dict(size=14))  # Reduced font size
        )
        st.plotly_chart(fig_imp)

    with col2:
        # ROC Curve with smaller height
        st.markdown('<div class="custom-title">ROC Curve</div>', unsafe_allow_html=True)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        roc_curve_fig = go.Figure()
        roc_curve_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
        roc_curve_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
        roc_curve_fig.update_layout(
            xaxis=dict(title="False Positive Rate", titlefont=dict(size=14)),  # Reduced font size
            yaxis=dict(title="True Positive Rate", titlefont=dict(size=14)),  # Reduced font size
            height=350  # Reduced plot height
        )
        st.plotly_chart(roc_curve_fig)

    # F1 Score vs Number of Predictors
    st.markdown('<div class="custom-title">F1 Score vs Number of Predictors</div>', unsafe_allow_html=True)

    # Use the actual results from fit_models
    result_df = pd.DataFrame(model_results)

    # Calculate the maximum F1 Score for each number of predictors
    f1_by_predictor_count = result_df.groupby('Number of Predictors')['F1 Score'].max().reset_index()

    # Plotting the data
    plt.figure(figsize=(4, 2.5))  # Reduced size for F1 Score plot
    plt.plot(f1_by_predictor_count['Number of Predictors'], f1_by_predictor_count['F1 Score'], marker='o')
    plt.xlabel('Number of Predictors', fontsize=10)  # Reduced font size for xlabel
    plt.ylabel('F1 Score', fontsize=10)  # Reduced font size for ylabel
    plt.title('F1 Score vs. Number of Predictors', fontsize=12)  # Reduced title size
    plt.grid(True)

    # Display the plot in the Streamlit app
    st.pyplot(plt)

    # Clear the figure to avoid overlapping plots if running multiple times
    plt.clf()