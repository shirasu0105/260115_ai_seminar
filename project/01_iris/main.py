import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#csvファイル読み込み
df = pd.read_csv(r'C:\Work\Python\260115_ai_seminar\project\01_iris\iris_dataset.csv')

#特徴量とラベルを分解
X = df.drop(columns=['target'])
y = df['target']

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)

# モデル生成
model = RandomForestClassifier(n_estimators=100, random_state=40)
model.fit(X_train, y_train)

# テストデータ予測作成
y_pred = model.predict(X_test)

# 評価結果表示
print("Accuracy Score:", accuracy_score(y_test,y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
