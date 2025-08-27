
import math
import numpy as np
import pandas as pd

# -----------------------------
# Survival / Life Expectancy
# -----------------------------
def survival_curve(age0=67, max_age=110, sex='male', a=None, b=0.09):
    """Return annual survival-from-retirement probabilities S(age).
    Uses a simple Gompertz hazard with default parameters chosen to roughly match
    AUS cohort life expectancy at age 67 (male ~20y, female ~23y).
    """
    if a is None:
        a = 3e-05 if sex=='male' else 2e-05  # baseline hazard at birth proxy
    ages = np.arange(age0, max_age+1)
    S=[1.0]
    s=1.0
    for age in ages[:-1]:
        H = (a/b)*(math.exp(b*(age+1))-math.exp(b*age))  # integrated hazard over the next year
        s *= math.exp(-H)
        S.append(s)
    return pd.Series(S, index=ages)

def survival_percentile_ages(age0=67, sex='male', percentiles=(0.5, 0.25), max_age=110):
    S = survival_curve(age0=age0, sex=sex, max_age=max_age)
    results = {}
    for p in percentiles:
        under = S[S < p]
        results[p] = None if len(under)==0 else int(under.index[0])
    return results

# -----------------------------
# Lifetime annuity (real / CPI-linked)
# -----------------------------
def annuity_payout_per_dollar(age0=67, sex='male', deferral_years=0, real_discount=0.015, max_age=110):
    """Payout per $1 premium for a real (inflation-linked) lifetime annuity.
    Assumes level real payments paid at the END of each year starting after the deferral period.
    Price is PV at purchase age0 with discount rate real_discount.
    """
    S = survival_curve(age0=age0, sex=sex, max_age=max_age)
    v = 1/(1+real_discount)
    pv = 0.0
    for t, age in enumerate(S.index.values):
        if t >= (deferral_years+1):  # first payment after deferral
            pv += (v**t) * S.iloc[t]
    return 0.0 if pv==0 else 1.0/pv

def annuity_income_schedule(purchase_amount, age0=67, sex='male', deferral_years=0, real_discount=0.015, max_age=110):
    rate = annuity_payout_per_dollar(age0, sex, deferral_years, real_discount, max_age)
    ages = np.arange(age0, max_age+1)
    income = np.zeros_like(ages, dtype=float)
    start_age = age0 + deferral_years + 1
    for i, age in enumerate(ages):
        if age >= start_age:
            income[i] = purchase_amount * rate
    return pd.Series(income, index=ages), rate

# -----------------------------
# Age Pension (simplified means test)
# -----------------------------
def deeming_income(financial_assets, deeming_threshold=60_400, deem_rate_low=0.0025, deem_rate_high=0.0225):
    low=min(financial_assets, deeming_threshold)
    high=max(0.0, financial_assets - deeming_threshold)
    return low*deem_rate_low + high*deem_rate_high

def age_pension_amount(assets, annuity_payment=0.0, annuity_purchase_price=0.0, age=67,
                       singles=True, homeowner=True, include_annuity_rules=True,
                       max_single=28_600, max_couple=43_100,
                       income_free_area_single=4_940, income_free_area_couple=8_736,
                       income_taper=0.5,
                       assets_threshold_single_home=301_750, assets_threshold_couple_home=451_500,
                       assets_taper_pa_per_1000=78.0,
                       deeming_threshold_single=60_400, deem_rate_low=0.0025, deem_rate_high=0.0225,
                       annuity_income_inclusion=0.60, annuity_asset_pct_before84=0.60, annuity_asset_pct_after84=0.30):
    """A lightweight approximation of the Australian Age Pension means test.
    All values are ANNUAL and roughly 2024-era defaults. Pass new parameters to update.
    Result is annual pension (real terms if your model is real).
    """
    max_pension = max_single if singles else max_couple
    # Income test (deeming on financial assets + annuity treatment)
    single_thresh = deeming_threshold_single
    couple_thresh = deeming_threshold_single*2
    deem_thresh = single_thresh if singles else couple_thresh
    assessable_income = deeming_income(assets, deeming_threshold=deem_thresh,
                                       deem_rate_low=deem_rate_low, deem_rate_high=deem_rate_high)
    if include_annuity_rules and annuity_payment>0:
        assessable_income += annuity_income_inclusion * annuity_payment
    free_area = income_free_area_single if singles else income_free_area_couple
    income_test_reduction = max(0.0, (assessable_income - free_area) * income_taper)
    pension_income_test = max(0.0, max_pension - income_test_reduction)

    # Assets test (annuity purchase price partially assessed)
    threshold = assets_threshold_single_home if singles else assets_threshold_couple_home
    annuity_asset = 0.0
    if include_annuity_rules and annuity_purchase_price>0:
        annuity_asset = annuity_asset_pct_before84 * annuity_purchase_price
        if age>=84:
            annuity_asset = annuity_asset_pct_after84 * annuity_purchase_price
    assessable_assets = assets + annuity_asset
    assets_excess = max(0.0, assessable_assets - threshold)
    assets_test_reduction = (assets_excess/1000.0) * assets_taper_pa_per_1000
    pension_assets_test = max(0.0, max_pension - assets_test_reduction)

    return min(pension_income_test, pension_assets_test)

# -----------------------------
# Main simulation
# -----------------------------
def simulate_retirement(starting_balance=600_000, retire_age=67, sex='male',
                        target_income=50_000, scenario='average',
                        annuity_alloc_pct=0.0, annuity_deferral_years=0, r_ann=0.015,
                        real_return_poor=0.01, real_return_avg=0.04, real_return_strong=0.07,
                        singles=True, homeowner=True, include_age_pension=True,
                        max_age=105):
    """Simulates annual income from Account-Based Pension (ABP), Age Pension, and a Deferred Lifetime Annuity.
    Returns a DataFrame with income components and end-year ABP balances, plus the annuity rate.
    """
    ages = list(range(retire_age, max_age+1))

    # Allocate to annuity at t=0
    annuity_premium = starting_balance * annuity_alloc_pct
    abp_balance = starting_balance - annuity_premium
    ann_income_series, ann_rate = annuity_income_schedule(annuity_premium, retire_age, sex,
                                                          annuity_deferral_years, r_ann, max_age)

    # Choose scenario return (real)
    r = {'poor': real_return_poor, 'average': real_return_avg, 'strong': real_return_strong}[scenario]

    abp_balances=[abp_balance]
    abp_income=[]
    age_pension=[]
    total_income=[]
    annuity_income=[]
    surv = survival_curve(retire_age, max_age, sex)

    bal = abp_balance
    for age in ages:
        a_income = float(ann_income_series.loc[age])
        if include_age_pension:
            pension = age_pension_amount(assets=bal, annuity_payment=a_income, annuity_purchase_price=annuity_premium,
                                         age=age, singles=singles, homeowner=homeowner)
        else:
            pension = 0.0
        # Draw from ABP to hit target (if possible)
        need = max(0.0, target_income - a_income - pension)
        draw = min(need, bal*(1+r))  # cannot draw more than after investment growth
        bal = bal*(1+r) - draw

        abp_balances.append(bal)
        abp_income.append(draw)
        age_pension.append(pension)
        annuity_income.append(a_income)
        total_income.append(draw + pension + a_income)

    df = pd.DataFrame({
        'age': ages,
        'abp_income': abp_income,
        'age_pension': age_pension,
        'annuity_income': annuity_income,
        'total_income': total_income,
        'abp_balance_end': abp_balances[1:],
        'survival_prob': surv.values[:len(ages)]
    })
    return df, ann_rate
