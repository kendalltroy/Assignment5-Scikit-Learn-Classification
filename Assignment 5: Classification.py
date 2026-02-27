# Imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#KNN Model
knn = KNeighborsClassifier(n_neighbors=10) #measures 10 neighbors in dataset
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

#prints model analysis
print('K-Nearest Neighbors Evaluation:')
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))

#Perceptron Model
perceptron = Perceptron(random_state=42)
perceptron.fit(X_train, y_train)
y_pred_perceptron = perceptron.predict(X_test)

#prints perceptron analysis
print('\nLinear Perceptron Evaluation:')
print("Accuracy:", accuracy_score(y_test, y_pred_perceptron))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_perceptron))
print("\nClassification Report:\n", classification_report(y_test, y_pred_perceptron))

#SVM
svm_model = SVC(kernel='rbf', random_state=42) #optimized SVC for nonlinear data
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

#prints SVM evaluation
print('\nSVM Model Evaluation:')
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

#K-Nearest Neighbors had the lowest overall accuracy and confusion matrix, so it can be assumed the model is not sufficient.
#The f-1 score remains strong for both the perceptron model and the SVM model, with only a 1% difference, making the models comparable.
#In the case of diagnosing cancer, it is best for there to be false positives rather than false negatives, as this would help catch more cases.
#The SVM model had the greatest overall accuracy (98%); however, the recall was 3% lower than the perceptron model.
#I would argue the perceptron model is overall better for this dataset, as the recall is much higher than the SVM model with little difference in accuracy, ensuring more positive cases are diagnosed correctly instead of being a false negative.