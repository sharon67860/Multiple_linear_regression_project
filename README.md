# Multiple_linear_regression_project
This project implements multiple linear regression using Python, focusing on preprocessing data (removing unnecessary columns), training a linear regression model, and evaluating its performance with metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).  

# Linear Regression Model Class
class MultipleLinearRegression:
    def _init_(self, data):
        self.data = data
        
    def preprocess_data(self):
        # Remove the first three columns
        self.data = self.data.iloc[:, 3:]
        # Split data into features and target (assuming last column is the target)
        self.X = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values
        return self.X, self.y
    
    def split_data(self, test_size=0.2):
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=0)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self):
        # Train the Linear Regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
    
    def evaluate_model(self):
        # Predict on test data
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy (R^2 score)
        train_accuracy = self.model.score(self.X_train, self.y_train)
        test_accuracy = self.model.score(self.X_test, self.y_test)
        
        # Calculate MSE, RMSE, and MAE
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        return train_accuracy, test_accuracy, mse, rmse, mae

# Usage example:
# Load the dataset
data = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')

# Create an object of the MultipleLinearRegression class
mlr = MultipleLinearRegression(data)

# Preprocess the data
X, y = mlr.preprocess_data()

# Split the data
X_train, X_test, y_train, y_test = mlr.split_data()

# Train the model
mlr.train_model()

# Evaluate the model
train_accuracy, test_accuracy, mse, rmse, mae = mlr.evaluate_model()

# Print the results
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
# output
train Accuracy : 0.9999999812764105


Train Loss(mean_Squared_error):  2.199001668389649


Train Loss(mean_absolute_error):  1.1765187604638958


Train Loss:(root_mean_square_error): 1.4829031217141762


test Accuracy : 0.9999999806028682
