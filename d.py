import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data: Replace these values with your actual model performance metrics
model_names = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'Support Vector Regression', 'Voting Regressor', 'AdaBoost']
r2_scores = [0.72, 0.68, 0.75, 0.78, 0.71, 0.76, 0.8042]
mae_scores = [0.045, 0.051, 0.042, 0.039, 0.047, 0.041, 0.0397]
mse_scores = [0.0065, 0.0072, 0.0059, 0.0054, 0.0067, 0.0057, 0.0053]

# Create a DataFrame for the metrics
df = pd.DataFrame({
    'Model': model_names,
    'R² Score': r2_scores,
    'MAE': mae_scores,
    'MSE': mse_scores
})

# Plot the R² score comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='R² Score', palette='coolwarm')
plt.title('Model Comparison: R² Score')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.show()

# Plot the MAE comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='MAE', palette='coolwarm')
plt.title('Model Comparison: Mean Absolute Error (MAE)')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.show()

# Plot the MSE comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Model', y='MSE', palette='coolwarm')
plt.title('Model Comparison: Mean Squared Error (MSE)')
plt.xlabel('Models')
plt.ylabel('MSE')
plt.xticks(rotation=45)
plt.show()