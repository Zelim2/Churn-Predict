import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder , OrdinalEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('train.csv')

# Числовые признаки
num_cols = [
    'ClientPeriod',
    'MonthlySpending',
]

# Категориальные признаки
cat_cols = [
    'Sex',
    'IsSeniorCitizen',
    'HasPartner',
    'HasChild',
    'HasPhoneService',
    'HasMultiplePhoneNumbers',
    'HasInternetService',
    'HasOnlineSecurityService',
    'HasOnlineBackup',
    'HasDeviceProtection',
    'HasTechSupportAccess',
    'HasOnlineTV',
    'HasMovieSubscription',
    'HasContractPhone',
    'IsBillingPaperless',
    'PaymentMethod'
]

feature_cols = num_cols + cat_cols
target_col = 'Churn'

def is_digit(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


d = dict()
for x in data['TotalSpent']:

    if not is_digit(x):
        if not x in d:
            d[x] = 1
        else:
            d[x] += 1

# есть строки где вместо NA стоят пропуски , можно просто избавиться от них
data = data[data['TotalSpent'] != ' ']
data['TotalSpent'] = data['TotalSpent'].apply(lambda x: float(x))

#закодируем категориальные данные
ord = OrdinalEncoder()
data[cat_cols] = ord.fit_transform(data[cat_cols])

# есть дисбаланс классов , поэтому я сделал следующее:
data_churn = data[data['Churn'] == 1]
data = pd.concat([data_churn , data] , axis = 0)

X_train = data[num_cols+cat_cols]
y_train = data[target_col]

scaler = StandardScaler()
log_model = LogisticRegression(C = 75 , penalty='l2').fit(scaler.fit_transform(X_train[num_cols]) , y_train)




X, y = data[num_cols + cat_cols], data[target_col]
X['sq1'] = X['MonthlySpending']**2
X['s1'] = X['ClientPeriod']*X['IsSeniorCitizen'] + X['MonthlySpending']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

best_model = GradientBoostingClassifier(
                                learning_rate=0.01,
                                n_estimators=5000,
                                max_depth=1,
                                verbose=False)
best_model.fit(X_train, y_train)

y_pred = best_model.predict_proba(X_test)[:, 1]

print(roc_auc_score(y_test, y_pred))