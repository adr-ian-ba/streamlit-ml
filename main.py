import streamlit as st
import pickle
import pandas as pd
import gdown

model_path = "best_rf_model.pkl"
url = "https://drive.google.com/uc?id=1Ji_Nc_VUH7SBbXQ7hNMyGL76_OqxHbnb"
gdown.download(url, model_path, quiet=False)

with open(model_path, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
encoders = saved["encoders"]
scaler = saved["scaler"]

st.markdown("### Test Case")

col1, col2 = st.columns(2)

with col1:
    if st.button("jadi book case"):
        test_case = {
            'no_of_adults': 2,
            'no_of_children': 0,
            'no_of_weekend_nights': 2,
            'no_of_week_nights': 3,
            'type_of_meal_plan': 'Meal Plan 1',
            'required_car_parking_space': 1,
            'room_type_reserved': 'Room_Type 1',
            'lead_time': 15,
            'arrival_month': 6,
            'market_segment_type': 'Online',
            'repeated_guest': 1,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 5,
            'avg_price_per_room': 150.0,
            'no_of_special_requests': 2
        }

        test_case['high_cancel_flag'] = 1 if test_case['no_of_previous_cancellations'] >= 5 else 0
        test_case['cancel_risk_score'] = (
            test_case['no_of_previous_cancellations'] * 2 +
            test_case['high_cancel_flag'] * 10 +
            (test_case['lead_time'] > 100) * 3 +
            (test_case['repeated_guest'] == 0) * 1
        )
        df = pd.DataFrame([test_case])
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        result = "Jadi Book" if prediction == 1 else "Cancel Book"
        st.success(f"# {result}")
        st.markdown("#### case settings:")
        for key, value in test_case.items():
            st.write(f"**{key}**: {value}")


with col2:
    if st.button("cancel book case"):
        test_case = {
            'no_of_adults': 1,
            'no_of_children': 0,
            'no_of_weekend_nights': 0,
            'no_of_week_nights': 1,
            'type_of_meal_plan': 'Meal Plan 1',
            'required_car_parking_space': 0,
            'room_type_reserved': 'Room_Type 4',
            'lead_time': 300,
            'arrival_month': 2,
            'market_segment_type': 'Online',
            'repeated_guest': 0,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 0,
            'avg_price_per_room': 60.0,
            'no_of_special_requests': 0
        }

        test_case['high_cancel_flag'] = 1 if test_case['no_of_previous_cancellations'] >= 5 else 0
        test_case['cancel_risk_score'] = (
            test_case['no_of_previous_cancellations'] * 2 +
            test_case['high_cancel_flag'] * 10 +
            (test_case['lead_time'] > 100) * 3 +
            (test_case['repeated_guest'] == 0) * 1
        )
        df = pd.DataFrame([test_case])
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)
        result = "Jadi Book" if prediction == 1 else "Cancel Book"
        st.warning(f"# {result}")
        st.markdown("#### case settings:")
        for key, value in test_case.items():
            st.write(f"**{key}**: {value}")



st.title("hotel predictor")

st.markdown("masukkand etail customer")


no_of_adults = st.slider("numbe rof adults", 1, 5, 2)
no_of_children = st.slider("number of children", 0, 5, 0)
no_of_weekend_nights = st.slider("weekend nights", 0, 5, 1)
no_of_week_nights = st.slider("weekday nights", 0, 10, 2)
type_of_meal_plan = st.selectbox("meal plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
required_car_parking_space = st.radio("car park needed?", [0, 1])
room_type_reserved = st.selectbox("room type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
lead_time = st.number_input("lead days (day before arival)", 0, 500, 100)
arrival_month = st.selectbox("arivel month", list(range(1, 13)))
market_segment_type = st.selectbox("market segment", ["Online", "Offline", "Corporate", "Complementary", "Aviation"])
repeated_guest = st.radio("repeated guest?", [0, 1])
no_of_previous_cancellations = st.slider("previous cancelation", 0, 10, 0)
no_of_previous_bookings_not_canceled = st.slider("previous success bookings (not canceled)", 0, 50, 0)
avg_price_per_room = st.number_input("room price", 0.0, 500.0, 100.0)
no_of_special_requests = st.slider("number of special request", 0, 5, 0)

# Submit
if st.button("Predict"):
    input_data = {
        'no_of_adults': no_of_adults,
        'no_of_children': no_of_children,
        'no_of_weekend_nights': no_of_weekend_nights,
        'no_of_week_nights': no_of_week_nights,
        'type_of_meal_plan': type_of_meal_plan,
        'required_car_parking_space': required_car_parking_space,
        'room_type_reserved': room_type_reserved,
        'lead_time': lead_time,
        'arrival_month': arrival_month,
        'market_segment_type': market_segment_type,
        'repeated_guest': repeated_guest,
        'no_of_previous_cancellations': no_of_previous_cancellations,
        'no_of_previous_bookings_not_canceled': no_of_previous_bookings_not_canceled,
        'avg_price_per_room': avg_price_per_room,
        'no_of_special_requests': no_of_special_requests
    }


    input_data['high_cancel_flag'] = 1 if input_data['no_of_previous_cancellations'] >= 5 else 0
    input_data['cancel_risk_score'] = (
        input_data['no_of_previous_cancellations'] * 2 +
        input_data['high_cancel_flag'] * 10 +
        (input_data['lead_time'] > 100) * 3 +
        (input_data['repeated_guest'] == 0) * 1
    )
    df = pd.DataFrame([input_data])


    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])


    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)
    result = "Jadi Book" if prediction == 1 else "Cancel Book"
    st.success(f"prediction: {result}")

