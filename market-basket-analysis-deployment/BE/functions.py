import pandas as pd

# Load rules once when starting the app
rules_df = pd.read_excel("final_result.xlsx")

# Convert string representation of tuple to actual tuple (in case needed)
rules_df['antecedents'] = rules_df['antecedents'].apply(eval)
rules_df['consequents'] = rules_df['consequents'].apply(eval)

def get_recommendations(user_items):
    recommendations = set()
    
    for _, row in rules_df.iterrows():
        if set(row['antecedents']).issubset(user_items):
            recommendations.update(row['consequents'])
    
    return list(recommendations - set(user_items))  # Exclude already purchased