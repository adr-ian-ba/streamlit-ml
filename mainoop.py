import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import pickle

# DataHandler Class
class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_data = None
        self.encoders = {}
        self.scaler = StandardScaler()

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df.drop(['Booking_ID', 'arrival_year', 'arrival_date'], axis=1, inplace=True)
        df['type_of_meal_plan'].fillna(df['type_of_meal_plan'].mode()[0], inplace=True)
        df['required_car_parking_space'].fillna(0, inplace=True)
        df['avg_price_per_room'].fillna(df['avg_price_per_room'].mean(), inplace=True)

        df['high_cancel_flag'] = df['no_of_previous_cancellations'].apply(lambda x: 1 if x >= 3 else 0)
        df['cancel_risk_score'] = (
            df['no_of_previous_cancellations'] * 2 +
            df['high_cancel_flag'] * 10 +
            (df['lead_time'] > 100).astype(int) * 3 +
            (df['repeated_guest'] == 0).astype(int) * 1
        )

        for col in ['lead_time', 'avg_price_per_room', 'no_of_previous_cancellations']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)

        cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'booking_status']
        for col in cat_cols:
            label = LabelEncoder()
            df[col] = label.fit_transform(df[col])
            self.encoders[col] = label

        self.output_data = df['booking_status']
        features = df.drop('booking_status', axis=1)
        self.input_df = pd.DataFrame(self.scaler.fit_transform(features), columns=features.columns)

    def get_data(self):
        return self.input_df, self.output_data, self.scaler, self.encoders



class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train, self.x_test, self.y_train, self.y_test = [None] * 4
        self.model = RandomForestClassifier(class_weight='balanced', max_depth=20, n_estimators=300, random_state=42)
        self.y_predict = None

    def split_data(self, test_size=0.2, random_state=42):
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(self.input_data, self.output_data)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X_resampled, y_resampled, test_size=test_size, random_state=random_state)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test)

    def createReport(self):
        print('classification res:')
        print(classification_report(self.y_test, self.y_predict))

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)

    def tuningParameter(self):
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'class_weight': ['balanced']
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='f1', cv=3)
        grid.fit(self.x_train, self.y_train)
        print("tuned:", grid.best_params_)
        print("best score:", grid.best_score_)
        self.model = grid.best_estimator_

    def save_model_to_file(self, filename, scaler, encoders):
        with open(filename, 'wb') as file:
            pickle.dump({'model': self.model, 'scaler': scaler, 'encoders': encoders}, file)



file_path = 'Dataset_B_hotel.csv'
data_handler = DataHandler(file_path)
data_handler.load_data()
input_df, output_df, scaler, encoders = data_handler.get_data()

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()

print("normal fit")
model_handler.train_model()
print("accuraccy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()

print("after tuning")
model_handler.tuningParameter()
model_handler.train_model()
print("accuraccy:", model_handler.evaluate_model())
model_handler.makePrediction()
model_handler.createReport()

model_handler.save_model_to_file('best_rf_model.pkl', scaler, encoders)