# imports
import numpy as np
from sklearn.datasets import fetch_california_housing # δεδομένα για πρόβλεψη
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression
from math import sqrt


# Φόρτωση δεδομένων
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)

# Λίστα για αποθήκευση RMSE τιμών του test set
test_rmse_list = []

# Εκτέλεση 20 πειραμάτων με διαφορετικά splits
for i in range(20):
    # Διαχωρισμός του dataset σε training (70%) και test (30%) set
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42 + i) # Κάθε φορά το random_state είναι διαφορετικό

    # Δημιουργία και εκπαίδευση του μοντέλου
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Αξιολόγηση στο test set και εκτίμηση σφάλματος του μοντέλου
    y_pred_test, mse_test = model.evaluate(X_test, y_test) # y εκτιμήτρια και MSE
    rmse_test = sqrt(mse_test) # Υπολογισμός τετραγωνικής ρίζας του MSE (RMSE)
    test_rmse_list.append(rmse_test)

    # Εκτύπωση αποτελέσματος ανά run
    print(f"Run {i + 1:2d}: Test RMSE = {rmse_test:.4f}")

print(f"\nTotal runs completed: {len(test_rmse_list)}")

# Υπολογισμός μέσης τιμής RMSE και τυπικής απόκλισης
mean_rmse = np.mean(test_rmse_list)
standard_rmse = np.std(test_rmse_list)

print(f"\nMean Test RMSE after 20 runs   : {mean_rmse:.4f}")
print(f"Standard Deviation of Test RMSE: {standard_rmse:.4f}")

