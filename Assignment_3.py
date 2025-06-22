from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/content/drive/MyDrive/IU2241230382/Global_Space_Exploration_Dataset.csv'
df = pd.read_csv(file_path)
df

df.head()

num_cols = ['Budget (in Billion $)', 'Success Rate (%)', 'Duration (in Days)']

# Mean
mean_values = df[num_cols].mean()
print("\nMean:\n", mean_values)

# Median
median_values = df[num_cols].median()
print("\nMedian:\n", median_values)

# Mode
mode_values = df[num_cols].mode().iloc[0]
print("\nMode:\n", mode_values)

# Standard Deviation
std_values = df[num_cols].std()
print("\nStandard Deviation:\n", std_values)

# Variance
var_values = df[num_cols].var()
print("\nVariance:\n", var_values)

df = df.dropna(subset=['Mission Type', 'Success Rate (%)'])

# Group by 'Mission Type' and calculate average success rate
success_rate = df.groupby('Mission Type')['Success Rate (%)'].mean()

# Print the results
print("Success Rate by Mission Type:\n")
for mission_type, rate in success_rate.items():
    print(f"{mission_type}: {rate:.2f}%")

df = df.dropna(subset=['Country', 'Success Rate (%)'])

# Group by Country and calculate average success rate
country_success = df.groupby('Country')['Success Rate (%)'].mean()

# Sort descending to get the highest
country_success_sorted = country_success.sort_values(ascending=False)

# Display top country
top_country = country_success_sorted.idxmax()
top_success_rate = country_success_sorted.max()

print(f"Country with Highest Average Success Rate:")
print(f"{top_country}: {top_success_rate:.2f}%\n")

print("Top 5 Countries by Success Rate:\n")
print(country_success_sorted.head())

top_5_countries = country_success_sorted.head(5)

# Plot
plt.figure(figsize=(8,5))
sns.barplot(x=top_5_countries.values, y=top_5_countries.index, palette='Blues_r')
plt.title('Top 5 Countries by Average Success Rate')
plt.xlabel('Average Success Rate (%)')
plt.ylabel('Country')

# Add value labels
for i, value in enumerate(top_5_countries.values):
    plt.text(value + 0.5, i, f"{value:.2f}%", va='center')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Country', y='Duration (in Days)', palette='Set2')
plt.title("Mission Duration by Country")
plt.xticks(rotation=45)
plt.ylabel("Duration (in Days)")
plt.show()

"""# Observations
- Most boxplots are relatively symmetric, but Russia and Germany might show slight left skew, indicating that some missions are much shorter compared to the rest.
- Most countries have similar median durations (around 160–190 days).
- Germany appears to have a slightly higher median compared to others.
- UAE and USA have somewhat lower medians.
- USA and Russia show a wider IQR — indicating more variability in mission duration
"""

sns.set(style="whitegrid")
# 1. Histogram of Success Rate
plt.figure(figsize=(8, 5))
sns.histplot(df['Success Rate (%)'], bins=20, kde=True, color='skyblue')
plt.title('Histogram of Success Rate (%)')
plt.xlabel('Success Rate (%)')
plt.ylabel('Frequency')
plt.show()
# 2. Boxplot of Budget by Mission Type
plt.figure(figsize=(8, 5))
sns.boxplot(x='Mission Type', y='Budget (in Billion $)', data=df, palette='pastel')
plt.title('Budget Distribution by Mission Type')
plt.xlabel('Mission Type')
plt.ylabel('Budget (in Billion $)')
plt.show()

"""# **Observations**
- **Some mission types (e.g., Human missions or Interplanetary) may have
significantly higher median budgets**.
- **Others (like CubeSats or Earth Observation) might be low-budget with tight IQRs.**
- **Wide boxes or long whiskers suggest high variability in budgets—likely driven by mission complexity, destination, or duration.**
"""

# Skewness
skewness = df[['Budget (in Billion $)', 'Success Rate (%)', 'Duration (in Days)']].skew()
print("\nSkewness:\n", skewness)

# Kurtosis
kurtosis = df[['Budget (in Billion $)', 'Success Rate (%)', 'Duration (in Days)']].kurt()
print("\nKurtosis:\n", kurtosis)

# Features (independent variables) and Target (dependent variable)
X = df[['Budget (in Billion $)', 'Duration (in Days)']]
y = df['Success Rate (%)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Linear Regression Evaluation:")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
print(f"R-squared (R² Score): {r2_score(y_test, y_pred):.2f}")

import statsmodels.api as sm
X_test_sm = sm.add_constant(X_test)  # Add intercept to training features
model_test_sm = sm.OLS(y_test, X_test_sm).fit()
print(model_test_sm.summary())

print("Final Analysis Summary\n")

# Descriptive Summary
print("1️. Descriptive Statistics:")
print("- Budget Mean:", round(df['Budget (in Billion $)'].mean(), 2), "Billion $")
print("- Success Rate Median:", df['Success Rate (%)'].median(), "%")
print("- Duration Std Dev:", round(df['Duration (in Days)'].std(), 2), "days\n")
print

# Skewness & Kurtosis Interpretation
print("2️. Data Distribution Insights:")
print("- Budget Skewness:", round(df['Budget (in Billion $)'].skew(), 2))
print("- Success Rate Kurtosis:", round(df['Success Rate (%)'].kurt(), 2))
print("- Interpretation: The data for Budget, Success Rate, and Duration is approximately symmetric (low skewness) but exhibits light tails(negative kurtosis), suggesting fewer extreme outliers than a normal distribution.\n")

# ML Summary
print("3️. Linear Regression Model:")
print("- Features: Budget, Duration")
print("- Target: Success Rate (%)")
print("- R² Score indicates how well model fits")
print("- Useful for estimating success rate from future mission plans\n")

# Key Insights
print("4️. Key Insights:")
print(f"- Germany has the highest average success rate of {top_success_rate:.2f}%")
print("- Countries with higher budgets don't always guarantee high success.")
print("- Some mission types (like manned) tend to have higher budgets.")
print("- Environmental impact & technology may influence success, worth exploring further.")
print("- Mission Type Success Rates\n" +  "   Manned Missions: ~75.23% success rate\n"+"   Unmanned Missions: ~74.73% success rate\n"+" Both types show nearly similar reliability.")

"""1. Descriptive Statistics
This helps us understand the basic numerical characteristics of the dataset.

➤ Mean
Average values:
- Budget: Tells the typical investment per mission.
- Success Rate: Gives an idea of the average mission success.
- Duration: How long missions last on average.

➤ Median
- The middle value in sorted data:

- More reliable than the mean when there are outliers (extreme values).

➤ Mode
- The most frequent value:
- Useful if one value (e.g., a common duration or success rate) appears much more often.

➤ Standard Deviation
- Measures how spread out the numbers are:
- High standard deviation in Budget or Success Rate means greater variability across missions.

➤ Variance
- Square of standard deviation:
- Also shows variability. High variance = inconsistent data.

2. Graphs (Visualizations)
- Histogram of Success Rate
 - Shows how frequently different success rates appear.
 - Helps identify distribution (normal, skewed, etc.).
 - You might see if most missions are generally successful or not.
- Boxplot of Budget by Mission Type
 - Visualizes distribution, median, and outliers.
 - Helps compare Manned vs Unmanned missions:
   - Are manned missions more expensive?
   - Is there more variability in costs?

3. Skewness & Kurtosis

- "Skewness"
 - Tells us about symmetry in data:
 - Positive skew: Tail on the right (many lower values, few large values
 - Negative skew: Tail on the left (many higher values, few low ones)
 - For example:
    - Budget might be right-skewed if a few missions have extremely high budgets.
- "Kurtosis"
  - Tells us about tailedness:
    - High kurtosis: More outliers, heavy tails.
    - Low kurtosis: Few outliers, flat distribution.

4. Machine Learning - Linear Regression
- You built a linear regression model to predict Success Rate using:
  - Budget
  - Duration
- What the model does:
 - Learns a linear relationship between budget/duration and mission success.
- Tells you:
  - How much success rate changes if you increase budget or duration.
  - How strong the prediction is using R² score.
- Evaluation Metrics:
 - Coefficients: Indicate how influential each feature is.
 - R² score: Value between 0–1; closer to 1 means good fit.
 - MSE (Mean Squared Error): Lower = better prediction.

5. Final Analysis Summary
 - Key Takeaways:
    - Germany has the highest average success rate of 76.25%
    - Higher budgets may not always lead to higher success rates.
    - Mission type plays a role in cost and perhaps success.
    - Some countries/collaborations tend to have more consistent or high-performing missions.
    - Environmental impact and technology might be potential predictors in future models.
    - Mission Type Success Rates
      - Manned Missions: ~75.23% success rate
      - Unmanned Missions: ~74.73% success rate
      -Both types show nearly similar reliability.
"""