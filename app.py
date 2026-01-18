# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load and clean data
# -----------------------------
@st.cache_data
def load_data():
    # Load CSV
    df = pd.read_csv("sales_1032_records.csv")
    
    # If the CSV has only one column with all data (common with Excel export), split it
    if df.shape[1] == 1:
        df = df.iloc[:, 0].str.split(r'\s{2,}', expand=True)
    
    # Assign correct column names if not automatically detected
    df.columns = ['Customer_ID','Age','Income','Credit_Score','Credit_Utilization','Missed_Payments',
                  'Delinquent_Account','Loan_Balance','Debt_to_Income_Ratio','Employment_Status',
                  'Account_Tenure','Credit_Card_Type','Location','Month_1','Month_2','Month_3',
                  'Month_4','Month_5','Month_6']
    
    # Convert numeric columns
    num_cols = ['Age','Income','Credit_Score','Credit_Utilization','Missed_Payments',
                'Delinquent_Account','Loan_Balance','Debt_to_Income_Ratio','Account_Tenure']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

df = load_data()

# -----------------------------
# Sidebar filters for dashboard
# -----------------------------
st.sidebar.header("Filters")
age_filter = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
income_filter = st.sidebar.slider("Income", int(df['Income'].min()), int(df['Income'].max()), (int(df['Income'].min()), int(df['Income'].max())))
credit_score_filter = st.sidebar.slider("Credit Score", int(df['Credit_Score'].min()), int(df['Credit_Score'].max()), (int(df['Credit_Score'].min()), int(df['Credit_Score'].max())))

# Apply filters
filtered_df = df[
    (df['Age'] >= age_filter[0]) & (df['Age'] <= age_filter[1]) &
    (df['Income'] >= income_filter[0]) & (df['Income'] <= income_filter[1]) &
    (df['Credit_Score'] >= credit_score_filter[0]) & (df['Credit_Score'] <= credit_score_filter[1])
]

# -----------------------------
# Dashboard
# -----------------------------
st.title("Velvet Crumbs Performance Dashboard")
st.write("Summary of selected customers")
st.dataframe(filtered_df)

# -----------------------------
# ML-based Recommender
# -----------------------------
st.header("Product Recommendations")

# Simulate products: assuming Month_1 to Month_6 indicate product usage
product_cols = ['Month_1','Month_2','Month_3','Month_4','Month_5','Month_6']

# Convert "Late/Missed/On-time" to numeric for similarity
usage_map = {'Late':1, 'Missed':1, 'On-time':0, np.nan:0}
usage_matrix = filtered_df[product_cols].replace(usage_map)

# Add Customer_ID as index
usage_matrix.index = filtered_df['Customer_ID']

# Compute similarity between customers
similarity = cosine_similarity(usage_matrix)
similarity_df = pd.DataFrame(similarity, index=usage_matrix.index, columns=usage_matrix.index)

# Select a customer
selected_customer = st.selectbox("Select Customer", usage_matrix.index.tolist())

# Find most similar customers
similar_customers = similarity_df[selected_customer].sort_values(ascending=False)
similar_customers = similar_customers.drop(selected_customer)  # exclude self

# Recommend products not used by selected customer
customer_usage = usage_matrix.loc[selected_customer]
products_to_recommend = customer_usage[customer_usage==0].index.tolist()

# If no products left, just recommend all products (fallback)
if len(products_to_recommend) == 0:
    products_to_recommend = usage_matrix.columns.tolist()

# Score recommendations based on similar customers
scores = {}
for product in products_to_recommend:
    score = usage_matrix.loc[similar_customers.index, product].mean()
    scores[product] = score

# Sort and pick top 3 recommendations
top_recommendations = sorted(scores.items(), key=lambda x: x[1])[:3]

# Show recommendations
st.subheader("Top Recommendations for Customer: " + selected_customer)
for i, (product, score) in enumerate(top_recommendations, 1):
    st.write(f"{i}. {product} (Predicted Interest Score: {1-score:.2f})")
