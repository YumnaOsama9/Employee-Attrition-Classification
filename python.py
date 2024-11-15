import streamlit as st
import numpy as np 
import streamlit as st
import numpy as np
import tensorflow as tf

# Caching the model loading process
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.keras")

# Load the model
loaded_model = load_model()

# Define mean and standard deviation values for numerical features

   # Define mean and standard deviation values for numerical features
mean_std_values = {
    "Age": {"mean": 38.56587469378167, "std": 12.079673030664623},
    "Years at Company": {"mean": 15.753901137622067, "std": 11.245981298833271},
    "Monthly Income": {"mean": 7302.397983153797, "std": 2151.457423002549},
    "Distance from Home": {"mean": 50.00765126346522, "std": 28.466458989438784},
    "Company Tenure": {"mean": 55.758414711903086, "std": 25.411089512757698}
}

# The rest of your Streamlit code remains the same.


# Standardize function
def standardize(value, feature):
    mean = mean_std_values[feature]["mean"]
    std = mean_std_values[feature]["std"]
    return (value - mean) / std

st.markdown("<h1 style='text-align: center; color: #FF6347; font-size: 42px;'>🔍Employee Retention Predictor🔍</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; color: #6A5ACD; font-size: 20px; font-family: "Comic Sans MS", sans-serif; line-height: 1.6;'>
Wondering if your team is here to stay or might be planning an exit? 🌟 Our smart, AI-driven tool is designed to provide you with insights into employee retention. 
Simply fill in the details below to receive a quick and easy analysis on employee turnover risk. Let's keep your best talent thriving!
</p>
""", unsafe_allow_html=True)


# Demographics section
with st.expander("📊 Demographics", expanded=True):
    age = st.slider("Age", min_value=18, max_value=100, value=25)
    years_at_company = st.slider("Years at Company", min_value=1, max_value=50, value=5)
    monthly_income = st.slider("Monthly Income", min_value=0, max_value=20000, step=100, value=5000)
    distance_from_home = st.slider("Distance from Home (in km)", min_value=0, max_value=100, value=10)
    company_tenure = st.slider("Company Tenure (in years)", min_value=0, max_value=200, value=2)

# Standardize numerical inputs
age1 = standardize(age, "Age")
years_at_company1 = standardize(years_at_company, "Years at Company")
monthly_income1 = standardize(monthly_income, "Monthly Income")
distance_from_home1 = standardize(distance_from_home, "Distance from Home")
company_tenure1 = standardize(company_tenure, "Company Tenure")

# Job Information section

with st.expander("🏢 Job Information", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female"])
    job_role = st.selectbox("Job Role", ["Healthcare", "Education", "Media", "Finance", "Technology"])
    work_life_balance = st.selectbox("Work-Life Balance", ["Excellent", "Good", "Fair", "Poor"])
    job_satisfaction = st.selectbox("Job Satisfaction", ["Very High", "High", "Medium", "Low"])
    performance_rating = st.selectbox("Performance Rating", ["High", "Average", "Low", "Below Average"])
    number_of_promotions = st.selectbox("Number of Promotions", [0, 1, 2, 3, 4])
    number_of_dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5, 6])
    education_level = st.selectbox("Education Level", ["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    overtime = st.selectbox("Overtime", ["Yes", "No"])
    job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    remote_work = st.selectbox("Remote Work", ["Yes", "No"])
    leadership_opportunities = st.selectbox("Leadership Opportunities", ["Yes", "No"])
    innovation_opportunities = st.selectbox("Innovation Opportunities", ["Yes", "No"])
    company_reputation = st.selectbox("Company Reputation", ["Excellent", "Good", "Fair", "Poor"])
    employee_recognition = st.selectbox("Employee Recognition", ["Very High", "High", "Medium", "Low"])

# Encode categorical features
gender_male = 1 if gender == "Male" else 0
job_role_finance = 1 if job_role == "Finance" else 0
job_role_healthcare = 1 if job_role == "Healthcare" else 0
job_role_media = 1 if job_role == "Media" else 0
job_role_technology = 1 if job_role == "Technology" else 0
work_life_balance_fair = 1 if work_life_balance == "Fair" else 0
work_life_balance_good = 1 if work_life_balance == "Good" else 0
work_life_balance_poor = 1 if work_life_balance == "Poor" else 0
job_satisfaction_low = 1 if job_satisfaction == "Low" else 0
job_satisfaction_medium = 1 if job_satisfaction == "Medium" else 0
job_satisfaction_very_high = 1 if job_satisfaction == "Very High" else 0
performance_rating_below_avg = 1 if performance_rating == "Below Average" else 0
performance_rating_high = 1 if performance_rating == "High" else 0
performance_rating_low = 1 if performance_rating == "Low" else 0
number_of_promotions_1 = 1 if number_of_promotions == 1 else 0
number_of_promotions_2 = 1 if number_of_promotions == 2 else 0
number_of_promotions_3 = 1 if number_of_promotions == 3 else 0
number_of_promotions_4 = 1 if number_of_promotions == 4 else 0
overtime_yes = 1 if overtime == "Yes" else 0
education_bachelors = 1 if education_level == "Bachelor's Degree" else 0
education_high_school = 1 if education_level == "High School" else 0
education_masters = 1 if education_level == "Master's Degree" else 0
education_phd = 1 if education_level == "PhD" else 0
marital_status_married = 1 if marital_status == "Married" else 0
marital_status_single = 1 if marital_status == "Single" else 0
dependents_1 = 1 if number_of_dependents == 1 else 0
dependents_2 = 1 if number_of_dependents == 2 else 0
dependents_3 = 1 if number_of_dependents == 3 else 0
dependents_4 = 1 if number_of_dependents == 4 else 0
dependents_5 = 1 if number_of_dependents == 5 else 0
dependents_6 = 1 if number_of_dependents == 6 else 0
job_level_mid = 1 if job_level == "Mid" else 0
job_level_senior = 1 if job_level == "Senior" else 0
company_size_medium = 1 if company_size == "Medium" else 0
company_size_small = 1 if company_size == "Small" else 0
remote_work_yes = 1 if remote_work == "Yes" else 0
leadership_opportunities_yes = 1 if leadership_opportunities == "Yes" else 0
innovation_opportunities_yes = 1 if innovation_opportunities == "Yes" else 0
company_reputation_fair = 1 if company_reputation == "Fair" else 0
company_reputation_good = 1 if company_reputation == "Good" else 0
company_reputation_poor = 1 if company_reputation == "Poor" else 0
employee_recognition_low = 1 if employee_recognition == "Low" else 0
employee_recognition_medium = 1 if employee_recognition == "Medium" else 0
employee_recognition_very_high = 1 if employee_recognition == "Very High" else 0

# Construct the input array with all 49 features
input_features = np.array([
    age1, years_at_company1, monthly_income1, distance_from_home1, company_tenure1,
    gender_male, job_role_finance, job_role_healthcare, job_role_media, job_role_technology,
    work_life_balance_fair, work_life_balance_good, work_life_balance_poor,
    job_satisfaction_low, job_satisfaction_medium, job_satisfaction_very_high,
    performance_rating_below_avg, performance_rating_high, performance_rating_low,
    number_of_promotions_1, number_of_promotions_2, number_of_promotions_3, number_of_promotions_4,
    overtime_yes, education_bachelors, education_high_school, education_masters, education_phd,
    marital_status_married, marital_status_single,
    dependents_1, dependents_2, dependents_3, dependents_4, dependents_5, dependents_6,
    job_level_mid, job_level_senior,
    company_size_medium, company_size_small,
    remote_work_yes, leadership_opportunities_yes, innovation_opportunities_yes,
    company_reputation_fair, company_reputation_good, company_reputation_poor,
    employee_recognition_low, employee_recognition_medium, employee_recognition_very_high
]).reshape(1, -1)  # Ensure input is (1, 49)



# Convert input_features to Tensor and predict
input_features = tf.convert_to_tensor(input_features, dtype=tf.float32)
pred = loaded_model.predict(input_features)

# Prediction button
if st.button("Predict"):
    result = "Stayed" if pred[0][0] >= 0.5 else "Left"
    if result == "Stayed":
        st.markdown("<h2 style='color: #4CAF50;'>Great News! 🎉</h2>", unsafe_allow_html=True)
        st.success("The prediction indicates that the employee is likely to stay with the company. 😊")
        st.balloons()
    else:
        st.markdown("<h2 style='color: #FF6347;'>Caution! ⚠️</h2>", unsafe_allow_html=True)
        st.error("The prediction suggests that the employee may leave the company. 😞 Consider reviewing engagement and retention strategies.")
