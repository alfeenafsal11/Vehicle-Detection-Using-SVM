from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from main import *hog_convertor,
# After collecting gradients and labels
X_train, X_test, y_train, y_test = train_test_split(gradients, labels, test_size=0.2, random_state=42)
svm = train_svm_model(X_train, y_train)

y_pred = svm.predict(hog_convertor(X_test))[1].ravel()
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
