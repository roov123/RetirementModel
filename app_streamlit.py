
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from retirement_model import simulate_retirement, survival_percentile_ages

st.set_page_config(page_title="Retirement Income Explorer (Python)", layout="wide")

st.sidebar.header("Inputs")
starting_balance = st.sidebar.number_input("Starting balance ($)", 0, 10_000_000, 600_000, step=10_000)
target_income = st.sidebar.number_input("Target annual income ($, real)", 0, 200_000, 50_000, step=1_000)
retirement_age = st.sidebar.slider("Retirement age", 55, 75, 67, 1)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
scenario = st.sidebar.selectbox("Market scenario", ["poor", "average", "strong"])
ann_pct = st.sidebar.slider("Deferred lifetime annuity allocation (%)", 0, 100, 0, 1) / 100.0
deferral = st.sidebar.slider("Annuity deferral period (years)", 0, 30, 0, 1)
include_pension = st.sidebar.checkbox("Include Age Pension", True)
single = st.sidebar.selectbox("Household", ["single","couple"])=="single"

df, ann_rate = simulate_retirement(starting_balance=starting_balance, retire_age=retirement_age, sex=sex,
                                   target_income=target_income, scenario=scenario,
                                   annuity_alloc_pct=ann_pct, annuity_deferral_years=deferral,
                                   singles=single, include_age_pension=include_pension)

st.title("Retirement Income by Age")
#st.caption(f"Annuity payout rate: {ann_rate:.2%} of premium per year (real).")

# Stacked bars of income components
fig, ax = plt.subplots(figsize=(7,3))
ax.bar(df['age'], df['age_pension'], label="Age Pension")
ax.bar(df['age'], df['annuity_income'], bottom=df['age_pension'], label="Deferred lifetime annuity")
ax.bar(df['age'], df['abp_income'], bottom=df['age_pension']+df['annuity_income'], label="Account-based pension")
ax.axhline(target_income, linestyle='--')


leg = ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.5), borderaxespad=0.)
fig.tight_layout()


# Survival markers
percs = survival_percentile_ages(age0=retirement_age, sex=sex, percentiles=(0.5,0.25), max_age=df['age'].max())
if percs.get(0.5):
    ax.axvline(percs[0.5], linestyle=':', linewidth=1,color='red')
if percs.get(0.25):
    ax.axvline(percs[0.25], linestyle=':', linewidth=1,color='red')

ax.set_xlabel("Age")
ax.set_ylabel("Annual income ($ real)")
ax.legend(loc="upper right")

st.pyplot(fig, use_container_width=False)


# ABP balance plot
fig2, ax2 = plt.subplots(figsize=(7,3))
ax2.plot(df['age'], df['abp_balance_end'])
ax2.set_xlabel("Age")
ax2.set_ylabel("ABP balance at year end ($ real)")
st.pyplot(fig2, use_container_width=False)

st.dataframe(df, use_container_width=True)
