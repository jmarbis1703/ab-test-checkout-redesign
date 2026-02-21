"""
Generate Synthetic A/B Test Data
Simulates a 21-day checkout-page redesign experiment with returning users:
- User level split randomization 
- Multi session journeys 
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
RNG = np.random.default_rng(SEED)

# Params
N_DAYS = 21
TARGET_TOTAL_SESSIONS = 76000

LATENT_SESSIONS_PER_USER = 0.57 
N_UNIQUE_USERS = int(TARGET_TOTAL_SESSIONS / LATENT_SESSIONS_PER_USER) 

START_DATE = pd.Timestamp('2024-09-02')

BASELINE_CVR = 0.032
TRUE_TREATMENT_LIFT = 0.004
BASELINE_AOV_MEAN = 68.0
BASELINE_AOV_STD = 32.0

DEVICE_PROBS = {'desktop': 0.42, 'mobile': 0.45, 'tablet': 0.13}
DEVICE_CVR_MULT = {'desktop': 1.15, 'mobile': 0.82, 'tablet': 1.05}
SOURCE_PROBS = {'organic': 0.35, 'paid_search': 0.28, 'social': 0.18, 'email': 0.12, 'direct': 0.07}

DOW_TRAFFIC_MULT = [1.0, 0.97, 0.95, 1.02, 1.08, 1.15, 1.05]

NOVELTY_PEAK = 0.003
NOVELTY_HALFLIFE = 4

def _novelty_effect(day_index):
    return NOVELTY_PEAK * np.exp(-np.log(2) * day_index / NOVELTY_HALFLIFE)

def generate_user_pool(n_users):
    print(f"Initializing pool of {n_users:,} latent users")
    user_ids = [f'U{i:07d}' for i in range(1, n_users + 1)]
    groups = RNG.choice(['control', 'treatment'], size=n_users, p=[0.5, 0.5])
    
    # Device Logic
    pref_devices = RNG.choice(list(DEVICE_PROBS.keys()), size=n_users, p=list(DEVICE_PROBS.values()))
    
    # Activity Scores: 
    # Beta(2, 5) creates a realistic e-commerce traffic skew 
    activity_scores = RNG.beta(a=2.0, b=5.0, size=n_users)
    activity_scores = activity_scores / activity_scores.mean() 
    
    return pd.DataFrame({
        'user_id': user_ids,
        'group': groups,
        'preferred_device': pref_devices,
        'activity_score': activity_scores
    })

def generate_dataset():
    users_df = generate_user_pool(N_UNIQUE_USERS)
    records = []
    
    #  Simulation Loop
    for day_idx in range(N_DAYS):
        date = START_DATE + pd.Timedelta(days=day_idx)
        dow = date.dayofweek
        base_sessions = (TARGET_TOTAL_SESSIONS / N_DAYS)
        daily_target = int(RNG.poisson(base_sessions * DOW_TRAFFIC_MULT[dow]))
        
        # Select Users for this Day
        probs = users_df['activity_score'].values.copy()
        probs /= probs.sum()
        
        daily_user_indices = RNG.choice(
            users_df.index, 
            size=daily_target, 
            p=probs, 
            replace=True
        )
        
        daily_users = users_df.loc[daily_user_indices].copy()
        
        # Session Details
        for _, user in daily_users.iterrows():
            if RNG.random() < 0.85:
                device = user['preferred_device']
            else:
                device = RNG.choice(list(DEVICE_PROBS.keys()))
            
            source = RNG.choice(list(SOURCE_PROBS.keys()), p=list(SOURCE_PROBS.values()))

            cvr = BASELINE_CVR * DEVICE_CVR_MULT[device]
            
            #  Treatment Effect
            if user['group'] == 'treatment':
                cvr += TRUE_TREATMENT_LIFT
                cvr += _novelty_effect(day_idx)
            
            converted = int(RNG.random() < cvr)

            revenue = 0.0
            if converted:
                aov = RNG.normal(BASELINE_AOV_MEAN, BASELINE_AOV_STD)
                revenue = round(max(aov, 5.0), 2)

            # Session duration: converters browse longer on average
            base_dur = RNG.lognormal(mean=4.8, sigma=0.7)
            if converted:
                base_dur *= 1.4
            session_duration = int(np.clip(base_dur, 10, 1800))
            
            # Timestamps
            hour_probs = np.array([0.01]*6 + [0.05]*6 + [0.08]*6 + [0.02]*6)
            hour_probs /= hour_probs.sum()
            hour = RNG.choice(24, p=hour_probs)
            ts = date + pd.Timedelta(hours=int(hour), minutes=RNG.integers(0, 60))
            
            records.append({
                'user_id': user['user_id'],
                'timestamp': ts,
                'group': user['group'],
                'device': device,
                'traffic_source': source,
                'converted': converted,
                'revenue': revenue,
                'session_duration_sec': session_duration
            })

    df = pd.DataFrame(records)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

if __name__ == '__main__':
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)
    
    df = generate_dataset()
    df.to_csv(out_dir / 'ab_test_data.csv', index=False)
    
    print(f"Generated {len(df):,} sessions.")
    obs_unique_users = df['user_id'].nunique()
    print(f"Unique Users (Observed): {obs_unique_users:,}")
    print(f"Avg Sessions/User: {len(df)/obs_unique_users:.2f}")
    
    multi_session_users = df['user_id'].value_counts()
    print(f"Users with >1 session: {(multi_session_users > 1).sum():,} ({(multi_session_users > 1).mean():.1%})")