import pickle
import pandas as pd


tes_input = {
    'no_of_adults': 2,
    'no_of_children': 1,
    'no_of_weekend_nights': 1,
    'no_of_week_nights': 2,
    'type_of_meal_plan': 'Meal Plan 1',
    'required_car_parking_space': 1,
    'room_type_reserved': 'Room_Type 1',
    'lead_time': 30,
    'arrival_month': 6,
    'market_segment_type': 'Online',
    'repeated_guest': 0,
    'no_of_previous_cancellations': 2,
    'no_of_previous_bookings_not_canceled': 2,
    'avg_price_per_room': 120.0,
    'no_of_special_requests': 1
}


tes_input['high_cancel_flag'] = 1 if tes_input['no_of_previous_cancellations'] >= 5 else 0
tes_input['cancel_risk_score'] = (
    tes_input['no_of_previous_cancellations'] * 2 +
    tes_input['high_cancel_flag'] * 10 +
    (tes_input['lead_time'] > 100) * 3 +
    (tes_input['repeated_guest'] == 0) * 1
)


with open("best_rf_model.pkl", "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
encoders = saved["encoders"]
scaler = saved["scaler"]

df = pd.DataFrame([tes_input])

for col, encoder in encoders.items():
    if col in df.columns:
        df[col] = encoder.transform(df[col])

df_scaled = scaler.transform(df)

prediction = model.predict(df_scaled)[0]
result = "Jadi Book" if prediction == 1 else "Cancel Book"

print(result)
