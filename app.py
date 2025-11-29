import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# New improved dashboard (English) and does NOT use MinMaxScaler or PCA.
@st.cache_data
def load_data(path="Data/customer_data_with_clusters.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'customer_data_with_clusters.csv' not found. Please run the clustering script first and put the file in 'Data/'.")
    st.stop()

# Ensure English UI
st.set_page_config(page_title="Customer Churn & Segmentation Insights", layout="wide")

# Validate and adapt to common dataset differences:
if 'Churn_Probability' not in df.columns and 'Churn' in df.columns:
    # If only binary churn column exists, treat it as a probability proxy
    df['Churn_Probability'] = df['Churn'].astype(float)
if 'Churn_Probability' not in df.columns:
    st.error("Missing 'Churn_Probability' (or 'Churn') column in the dataset.")
    st.stop()

if 'Cluster' not in df.columns:
    st.error("Missing 'Cluster' column in the dataset.")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
clusters = sorted(df['Cluster'].unique())
selected_clusters = st.sidebar.multiselect("Select cluster(s):", options=clusters, default=clusters)

contract_values = sorted(df['Contract'].dropna().unique()) if 'Contract' in df.columns else []
internet_values = sorted(df['InternetService'].dropna().unique()) if 'InternetService' in df.columns else []
payment_values = sorted(df['PaymentMethod'].dropna().unique()) if 'PaymentMethod' in df.columns else []

contract_filter = st.sidebar.multiselect("Contract type:", options=contract_values, default=contract_values)
internet_filter = st.sidebar.multiselect("Internet Service:", options=internet_values, default=internet_values)
payment_filter = st.sidebar.multiselect("Payment method:", options=payment_values, default=payment_values)

min_churn, max_churn = 0.0, 1.0
churn_range = st.sidebar.slider("Churn probability range:", min_value=float(min_churn), max_value=float(max_churn), value=(float(min_churn), float(max_churn)), step=0.01)

# Slice dataset based on filters
df_filtered = df[df['Cluster'].isin(selected_clusters)].copy()
if contract_values:
    df_filtered = df_filtered[df_filtered['Contract'].isin(contract_filter)]
if internet_values:
    df_filtered = df_filtered[df_filtered['InternetService'].isin(internet_filter)]
if payment_values:
    df_filtered = df_filtered[df_filtered['PaymentMethod'].isin(payment_filter)]

df_filtered = df_filtered[(df_filtered['Churn_Probability'] >= churn_range[0]) & (df_filtered['Churn_Probability'] <= churn_range[1])]

# Top bar KPIs
st.title("Customer Churn & Segmentation Dashboard (New Layout)")
st.markdown("English-only UI. No MinMaxScaler or PCA used.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Selected Customers", f"{len(df_filtered):,}")
col2.metric("Average Churn Probability", f"{df_filtered['Churn_Probability'].mean():.2%}")
if 'MonthlyCharges' in df_filtered.columns:
    col3.metric("Avg Monthly Revenue", f"${df_filtered['MonthlyCharges'].mean():.2f}")
else:
    col3.metric("Avg Monthly Revenue", "N/A")
if 'tenure' in df_filtered.columns:
    col4.metric("Avg Tenure (months)", f"{df_filtered['tenure'].mean():.1f}")
else:
    col4.metric("Avg Tenure (months)", "N/A")

st.markdown("---")

# Cluster summary table with rankings and lift
def cluster_summary(df_in):
    num_cols = df_in.select_dtypes(include=[np.number]).columns.tolist()
    summary = df_in.groupby('Cluster').agg(
        cluster_size=('Cluster', 'size'),
        avg_churn=('Churn_Probability', 'mean'),
        predicted_churn=('Churn_Probability', 'sum')
    )
    if 'MonthlyCharges' in df_in.columns:
        summary['avg_monthly'] = df_in.groupby('Cluster')['MonthlyCharges'].mean()
    if 'tenure' in df_in.columns:
        summary['avg_tenure'] = df_in.groupby('Cluster')['tenure'].mean()
    # Churn lift vs global
    global_churn = df_in['Churn_Probability'].mean()
    summary['churn_lift'] = summary['avg_churn'] - global_churn
    summary = summary.sort_values('avg_churn', ascending=False)
    return summary, global_churn

summary_df, global_churn = cluster_summary(df_filtered)
st.header("Cluster Summary & Priority Ranking")
st.write("Clusters sorted by average churn probability. 'Churn lift' is cluster avg minus overall avg.")

# highlight the churn column
st.dataframe(summary_df.style.format({"avg_churn": "{:.2%}", "predicted_churn": "{:.0f}", "avg_monthly": "${:.2f}", "avg_tenure": "{:.1f}", "churn_lift": "{:.2%}"}), use_container_width=True)

# Suggested actions and priority score
st.subheader("Retention Priority & Suggested Actions")
def retention_priority_score(row):
    # Priority ~ churn probability * monthly revenue * log(cluster size)
    monthly = row.get('avg_monthly', np.nan)
    if np.isnan(monthly):
        monthly = 1.0
    score = row['avg_churn'] * monthly * np.log1p(row['cluster_size'])
    return score

if not summary_df.empty:
    summary_df['priority_score'] = summary_df.apply(retention_priority_score, axis=1)
    summary_df = summary_df.sort_values('priority_score', ascending=False)
    st.write("Clusters sorted by a priority score (churn * revenue * cluster size adjustment).")
    st.dataframe(summary_df[['cluster_size', 'avg_churn', 'avg_monthly', 'priority_score']].style.format({"avg_churn":"{:.2%}", "avg_monthly":"${:.2f}", "priority_score":"{:.2f}"}), use_container_width=True)

# Strategy generation based on cluster metrics
def strategy_for_cluster(idx, row):
    churn = row['avg_churn']
    size = int(row['cluster_size'])
    avg_month = row.get('avg_monthly', 0.0)
    actions = []
    if churn >= 0.6:
        actions.append("Immediate retention campaigns and save-offers")
    elif churn >= 0.35:
        actions.append("Targeted offers, loyalty incentives")
    else:
        actions.append("Upsell/expansion and premium offers")
    if avg_month >= 80:
        actions.append("High-ARPU prioritization: VIP retention & premium bundles")
    if size < 200 and churn >= 0.5:
        actions.append("Personalized outreach and phone retention for each customer")
    if 'Contract' in df_filtered.columns:
        # show contract mode
        preferred_contract = df_filtered[df_filtered['Cluster']==idx]['Contract'].mode().iloc[0] if not df_filtered[df_filtered['Cluster']==idx]['Contract'].mode().empty else None
        if preferred_contract and preferred_contract in ['Month-to-month', 'month-to-month']:
            actions.append("Offer longer-term contract discounts to reduce churn")
    return ". ".join(actions)

strategy_map = {idx: strategy_for_cluster(idx, row) for idx, row in summary_df.iterrows()}
strat_df = pd.DataFrame.from_dict(strategy_map, orient='index', columns=['Suggested Actions']).reset_index().rename(columns={'index':'Cluster'})
st.table(strat_df.set_index('Cluster'))

st.markdown("---")
st.header("Visual Insights")

# 1. Cluster sizes bar chart + avg churn as line
cluster_bar_data = summary_df.reset_index()
fig = go.Figure()
fig.add_trace(go.Bar(x=cluster_bar_data['Cluster'], y=cluster_bar_data['cluster_size'], name='Cluster Size'))
fig.add_trace(go.Scatter(x=cluster_bar_data['Cluster'], y=cluster_bar_data['avg_churn'], mode='lines+markers', name='Avg Churn', marker=dict(color='red'), yaxis='y2'))
fig.update_layout(title="Cluster Size and Average Churn", yaxis_title="Size", yaxis2=dict(overlaying='y', side='right', title='Avg Churn'))
st.plotly_chart(fig, use_container_width=True)

# 2. Churn distribution (violin or histogram)
# st.subheader("Churn Probability Distribution by Cluster")
# fig_violin = px.violin(df_filtered, x='Cluster', y='Churn_Probability', color='Cluster', box=True, points='all', title='Churn Probability Distribution per Cluster')
# st.plotly_chart(fig_violin, use_container_width=True)

# 3. Monthly Charges vs Churn scatter with cluster color
if 'MonthlyCharges' in df_filtered.columns:
    st.subheader("Monthly Charges vs Churn Probability")
    fig_scatter = px.scatter(df_filtered, x='MonthlyCharges', y='Churn_Probability', color='Cluster', hover_data=['tenure', 'TotalCharges'], title='Monthly Charges vs Churn')
    st.plotly_chart(fig_scatter, use_container_width=True)

# 4. Heatmap: Contract x PaymentMethod average churn per cluster (shows where churn concentrates)
if 'Contract' in df_filtered.columns and 'PaymentMethod' in df_filtered.columns:
    st.subheader("Average Churn: Contract vs Payment Method (sample per cluster)")

    # Choose up to 4 clusters and compute per-cluster pivot tables
    clusters_to_plot = list(df_filtered['Cluster'].unique())[:4]
    pivots = []
    for c in clusters_to_plot:
        sub = df_filtered[df_filtered['Cluster'] == c]
        pivot = (
            sub.groupby(['Contract', 'PaymentMethod'])['Churn_Probability']
            .mean()
            .unstack(fill_value=np.nan)
        )
        pivots.append((c, pivot))

    # get global color scale bounds across pivots (skip NaNs)
    all_vals = np.concatenate([p.values.flatten() for _, p in pivots]) if pivots else np.array([0.0])
    finite = np.isfinite(all_vals)
    if finite.any():
        vmin = float(np.nanmin(all_vals[finite]))
        vmax = float(np.nanmax(all_vals[finite]))
    else:
        vmin, vmax = 0.0, 1.0

    # Layout: 2x2 grid using two rows of two columns
    rows = [pivots[:2], pivots[2:4]]
    for row in rows:
        cols = st.columns(2)
        for i in range(2):
            if i < len(row):
                cluster_id, pivot = row[i]
                if pivot.shape[0] == 0 or pivot.shape[1] == 0:
                    cols[i].write(f"Cluster {cluster_id}: Not enough data for heatmap")
                    continue
                # create heatmap with unified zmin/zmax, show text values and larger height
                fig_heat = px.imshow(
                    pivot,
                    color_continuous_scale='RdYlBu_r',
                    zmin=vmin,
                    zmax=vmax,
                    text_auto='.2f',
                    labels=dict(x='PaymentMethod', y='Contract', color='Avg Churn'),
                    title=f"Cluster {cluster_id} - Avg churn",
                    aspect='auto',
                    height=420,
                )
                # style improvements
                fig_heat.update_xaxes(tickangle=45, tickfont=dict(size=11), automargin=True)
                fig_heat.update_yaxes(tickfont=dict(size=11), automargin=True)
                fig_heat.update_traces(colorbar=dict(thickness=20, tickformat='.0%'), hovertemplate=None)
                fig_heat.update_layout(margin=dict(l=30, r=10, t=40, b=80))
                cols[i].plotly_chart(fig_heat, use_container_width=True)
            else:
                # empty placeholder for layout balance
                cols[i].empty()

# 5. Feature-difference view: per-cluster differences from global mean for numeric features
st.subheader("Cluster Feature Differences vs Global Mean")
numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['Cluster']]  # keep Churn_Probability included
global_means = df_filtered[numeric_cols].mean()
feature_diffs = df_filtered.groupby('Cluster')[numeric_cols].mean().subtract(global_means)
# Display top 3 features that most distinguish each cluster (by absolute diff)
top_features = {}
for cl in feature_diffs.index:
    row = feature_diffs.loc[cl].abs().sort_values(ascending=False).head(3)
    top_features[cl] = ", ".join([f"{feat} ({feature_diffs.loc[cl, feat]:+.2f})" for feat in row.index])
feature_diff_df = pd.DataFrame.from_dict(top_features, orient='index', columns=['Top Distinguishing Features']).reset_index().rename(columns={'index':'Cluster'})
st.table(feature_diff_df.set_index('Cluster'))

# 6. Churn vs Tenure buckets — to see early churners
if 'tenure' in df_filtered.columns:
    st.subheader("Churn Rate by Tenure Bucket")
    bins = [0,1,3,6,12,24,48,72,100]
    # create the interval buckets (keep as categorical for ordering), then convert to string label for Plotly
    df_filtered['tenure_bucket'] = pd.cut(df_filtered['tenure'].fillna(0), bins, include_lowest=True)
    churn_by_bucket = (
        df_filtered.groupby('tenure_bucket', observed=True)['Churn_Probability']
        .mean()
        .reset_index()
    )
    # convert Interval to readable string to avoid JSON serialization error
    churn_by_bucket['tenure_bucket_label'] = churn_by_bucket['tenure_bucket'].astype(str)
    fig_bucket = px.bar(
        churn_by_bucket,
        x='tenure_bucket_label',
        y='Churn_Probability',
        title='Avg Churn by Tenure Bucket'
    )
    fig_bucket.update_xaxes(type='category')
    st.plotly_chart(fig_bucket, use_container_width=True)

# 7. Top customers to target (by churn probability * monthly charges)
st.markdown("---")
st.header("Top Customers to Target (High Risk x Value)")
id_col_candidates = ['customerID', 'customer_id', 'customerId', 'ID', 'id']
id_col = next((c for c in id_col_candidates if c in df_filtered.columns), None)
# compute target_score
if 'MonthlyCharges' in df_filtered.columns:
    df_filtered['target_score'] = df_filtered['Churn_Probability'] * df_filtered['MonthlyCharges']
else:
    df_filtered['target_score'] = df_filtered['Churn_Probability']

top_n = st.number_input("Show top N customers:", min_value=5, max_value=200, value=20)
top_customers = df_filtered.sort_values('target_score', ascending=False).head(int(top_n))
cols_to_show = [id_col] if id_col else []
cols_to_show += ['Cluster', 'Churn_Probability', 'MonthlyCharges', 'tenure', 'target_score']
cols_to_show = [c for c in cols_to_show if c in top_customers.columns]
st.dataframe(top_customers[cols_to_show].fillna("N/A").reset_index(drop=True), use_container_width=True)

# download selected data
st.markdown("---")
if len(df_filtered) > 0:
    csv = df_filtered.to_csv(index=False)
    st.download_button("Download selected data as CSV", csv, file_name='churn_selection.csv', mime='text/csv')
else:
    st.write("No data available for the selected filters.")

# Footer / quick tips
st.markdown("---")
st.write("Tips:")
st.write("- Use the filters in the sidebar to focus on a subset of clusters or contract types.")
st.write("- Priority score is a heuristic.")
# st.write("- For more advanced ML explanations, consider storing feature importances per prediction and surfacing top features per customer.")