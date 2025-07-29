import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Calculate the accuracy and f1 score of a model
def calc_metrics(predictions, y_test):
    accuracy = np.mean(predictions == y_test)
    f1_metric = f1_score(y_test, predictions)

    print('Accuracy of Model: {:.2f}%'.format(100 * accuracy))
    print('F1 Score of Model: {:.4f}'.format(f1_metric))




df = pd.read_csv('Bird_strikes.csv')

# Create binary variable (1 for 'Caused damage', 0 for 'No damage')
df['Damage'] = (df['Damage'] == 'Caused damage').astype(int)

# Select features 
features = ['NumberStruckActual', 'Altitude', 'WildlifeSize', 
           'FlightPhase', 'ConditionsSky', 'PilotWarned', 
           'IsAircraftLarge?']

X = df[features].copy()


categorical_features = ['WildlifeSize', 'FlightPhase', 'ConditionsSky', 
                       'PilotWarned', 'IsAircraftLarge?']

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])


numerical_features = ['NumberStruckActual', 'Altitude']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, df['Damage'], test_size=0.2, random_state=42
)


#############   baseline   ############# 
baseline_pred = [0 for _ in range(len(y_test))]
calc_metrics(baseline_pred, y_test)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, baseline_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Damage', 'Caused Damage'],
            yticklabels=['No Damage', 'Caused Damage'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
############# ############## ############# 


############ Evaluate Standard Machine Learning Methods ##############
#### LR
lr = LogisticRegressionCV(Cs= 20, cv = 3, scoring = 'f1', 
                          penalty = 'l2', random_state = 42)
lr.fit(X_train, y_train)

print("LR result:")
lr_pred = lr.predict(X_test)
calc_metrics(lr_pred, y_test)

#### RF
rf = RandomForestClassifier(n_estimators=100, random_state = 42)
rf.fit(X_train, y_train)

print("RF result:")
rf_pred = rf.predict(X_test)
calc_metrics(rf_pred, y_test)

#### SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svc = SVC(C = 10, gamma = 0.001, probability=True,
          random_state = 42)
svc.fit(X_scaled, y_train)

print("SVC result:")
svc_pred = svc.predict(X_test_scaled)
calc_metrics(svc_pred, y_test)
############ ############## ############## ############## ##############





### PyMC Model
with pm.Model() as model:
    X_shared = pm.Data("X_shared", X_train)
    # Priors for unknown model parameters
    intercept = pm.Normal('intercept', 0, tau=0.1)
    betas = pm.Normal('betas', 0, tau=0.01, shape=X_train.shape[1])
    mu = intercept + pm.math.dot(X_shared, betas)
    p = pm.math.sigmoid(mu)
    y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train, shape=mu.shape)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)

# Print summary of the model
az.summary(trace, var_names=['intercept', 'betas'], hdi_prob=0.95)
az.plot_trace(trace)
az.plot_posterior(trace)
model.to_graphviz()


X_test.shape  # 5086,7


with model:
    pm.set_data({"X_shared": X_test})
    posterior_pred = pm.sample_posterior_predictive(trace, return_inferencedata=False)



y_pred_dist = posterior_pred['y_obs']  # Shape: (chains, draws, test_size)

y_pred_mean = y_pred_dist.reshape(-1, y_pred_dist.shape[-1]).mean(axis=0)  # Mean across all chains and draws

y_pred = (y_pred_mean > 0.5).astype(int)


print("Shape of y_pred:", y_pred.shape)
print("Shape of y_test:", y_test.shape)
assert y_pred.shape[0] == y_test.shape[0], "Mismatch between predictions and test labels."

# Model Performance
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Confusion Matrix 
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")
print(f"f1 score: {f1:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Damage', 'Caused Damage'],
            yticklabels=['No Damage', 'Caused Damage'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

