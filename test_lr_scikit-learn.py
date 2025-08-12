# imports
import numpy as np
from sklearn.datasets import fetch_california_housing # δεδομένα για πρόβλεψη
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # γραμμική παλινδρόμηση του sklearn
from sklearn.metrics import mean_squared_error


# Φόρτωση δεδομένων και διαχωρισμός σε χαρακτηριστικά (X) και στόχο (y)
X, y = fetch_california_housing(return_X_y=True)

rmse_list = [] # Λίστα για αποθήκευση RMSE τιμών του test set

for run in range(20):

    # Διαχωρισμός του dataset σε training (70%) και test (30%) set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=run
    )

    # Εκπαίδευση με sklearn
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Πρόβλεψη και υπολογισμός RMSE
    y_predict = model.predict(X_test) # y εκτιμήτρια
    rmse = np.sqrt(mean_squared_error(y_test, y_predict)) # Υπολογισμός τετραγωνικής ρίζας του MSE (RMSE)
    rmse_list.append(rmse)

    # Εκτύπωση αποτελέσματος ανά run
    print(f"Run {run+1:2d}: Test RMSE = {rmse:.4f}")

print(f"\nTotal runs completed: {len(rmse_list)}")

# Υπολογισμός μέσης τιμής RMSE και τυπικής απόκλισης
mean_rmse = np.mean(rmse_list)
standard_rmse = np.std(rmse_list)

print(f"\nMean Test RMSE after 20 runs   : {mean_rmse:.4f}")
print(f"Standard Deviation of Test RMSE: {standard_rmse:.4f}")

