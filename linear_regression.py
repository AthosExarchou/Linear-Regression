# imports
import numpy as np


# Κλάση για την υλοποίηση της γραμμικής παλινδρόμησης
class LinearRegression:

    def __init__(self):
        self.w = None # Βάρη (συντελεστές) του μοντέλου
        self.b = None # Όρος μεροληψίας (bias)


    # Εκπαίδευση του μοντέλου με βάση τα δεδομένα X και y
    def fit(self, X, y):
        """
        X: np.ndarray με σχήμα (N, p), όπου N δείγματα και p χαρακτηριστικά
        y: np.ndarray με σχήμα (N, 1), τιμές στόχου
        """

        # Έλεγχος εγκυρότητας εισόδου
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Τα X και y πρέπει να είναι numpy arrays.")

        # Έλεγχος διαστάσεων των εισόδων
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("Το X πρέπει να είναι 2D (N x p) και το y πρέπει να είναι 2D (N x 1).")

        N, p = X.shape
        if y.shape[0] != N or y.shape[1] != 1:
            raise ValueError("Το y πρέπει να έχει σχήμα (N, 1), συμβατό με το X.")

        # Δημιουργία επαυξημένου πίνακα X (X_augmented) προσθέτοντας στήλη με 1 για το bias
        X_augmented = np.hstack([X, np.ones((N, 1))]) # Σχήμα (N, p+1)

        # Υπολογισμός κανονικών εξισώσεων θ = (X^T X)^-1 X^T y
        XTX = np.dot(X_augmented.T, X_augmented)
        XTy = np.dot(X_augmented.T, y)

        try:
            thita = np.dot(np.linalg.inv(XTX), XTy) # Σχήμα (p+1, 1)
        except np.linalg.LinAlgError:
            raise ValueError("Ο πίνακας X^T X δεν είναι αντιστρέψιμος.")

        # Αποθήκευση βαρών και bias
        self.w = thita[:-1].reshape(-1) # Από 2D σε 1D πίνακα
        self.b = thita[-1, 0] # Το τελευταίο στοιχείο (bias)


    # Υπολογίζει τις τιμές πρόβλεψης του μοντέλου για νέο πίνακα χαρακτηριστικών X
    def predict(self, X):
        """
        X: np.ndarray, σχήμα (N, p)
        """

        # Έλεγχος αν το μοντέλο έχει εκπαιδευτεί
        if self.w is None or self.b is None:
            # Αν η fit() δεν έχει κληθεί, τότε το μοντέλο δεν έχει "μάθει" τίποτα ακόμα
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Πρώτα πρέπει να κληθεί η fit().")

        # Έλεγχος διαστάσεων εισόδου
        if not isinstance(X, np.ndarray):
            raise ValueError("Το X πρέπει να είναι numpy array.")

        if X.shape[1] != len(self.w):
            raise ValueError(f"Το X έχει {X.shape[1]} χαρακτηριστικά, όμως το μοντέλο έχει εκπαιδευτεί με {len(self.w)}.")

        # Υπολογισμός προβλέψεων
        y_predict = np.dot(X, self.w) + self.b # εκτιμήτρια y
        return y_predict.reshape(-1, 1) # Επιστρέφει τις προβλέψεις με σχήμα (N, 1)

    # Αξιολόγηση του μοντέλου και υπολογισμός του μέσου τετραγωνικού σφάλματος MSE
    def evaluate(self, X, y):
        """
        X: np.ndarray (N, p) με δείγματα
        y: np.ndarray (N, 1) με πραγματικές τιμές
        """

        # Έλεγχος αν το μοντέλο έχει εκπαιδευτεί
        if self.w is None or self.b is None:
            # Αν η fit() δεν έχει κληθεί, τότε το μοντέλο δεν έχει "μάθει" τίποτα ακόμα
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί. Πρώτα πρέπει να κληθεί η fit().")

        # Έλεγχος εγκυρότητας εισόδου
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Το X και το y πρέπει να είναι numpy arrays.")

        if y.ndim != 2 or y.shape[1] != 1:
            raise ValueError("Το y πρέπει να έχει σχήμα (N, 1).")

        if X.shape[0] != y.shape[0]:
            raise ValueError("Το X και το y πρέπει να έχουν ίδιο αριθμό γραμμών (δείγματα).")

        # Υπολογισμός προβλέψεων
        y_predict = self.predict(X) # εκτιμήτρια y

        # Υπολογισμός μέσου τετραγωνικού σφάλματος
        errors = y_predict - y # Σχήμα (N, 1)
        mse = np.dot(errors.T, errors) / X.shape[0] # (1, 1)
        mse_value = mse.item() # Μετατροπή σε float

        return y_predict, mse_value

