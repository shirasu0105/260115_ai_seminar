import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# csvファイル読み込み
df = pd.read_csv(r'C:\Work\Python\260115_ai_seminar\project\02_wine\wine_dataset.csv')

# 特徴量とラベルを分離
X = df.drop(columns=['target'])
y = df['target']

# データ分割の乱数のみ可変
seed_list = [7 + 13 * i for i in range(100)]
results = []
analysis_seed = 384

for split_seed in seed_list:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=split_seed, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=40, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((split_seed, accuracy))

    if split_seed == analysis_seed:
        print('--- Misclassification analysis (seed=384) ---')
        print('confusion_matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('classification_report:')
        print(classification_report(y_test, y_pred))

accuracies = [accuracy for _, accuracy in results]
best_seed, best_accuracy = max(results, key=lambda x: x[1])
worst_seed, worst_accuracy = min(results, key=lambda x: x[1])

print('runs:', len(seed_list))
print('mean:', f'{statistics.mean(accuracies):.4f}')
print('std:', f'{statistics.pstdev(accuracies):.4f}')
print('min:', f'{worst_accuracy:.4f}', f'(seed={worst_seed})')
print('max:', f'{best_accuracy:.4f}', f'(seed={best_seed})')
