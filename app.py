import streamlit as st
import pandas as pd
from datetime import date
import pickle
import catboost
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from PIL import Image
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Style for labels */
        .stTextInput > label,
        .stSelectbox > label,
        .stNumberInput > label,
        .stTextArea > label {{
            color: yellow; /* Set label color to yellow */
        }}

        /* Style for the title */
        .css-1e8b04i {{
            color: yellow; /* Set title color to yellow */
            font-size: 2em; /* Adjust font size if needed */
        }}

        /* Style for the prediction result */
        .prediction-result {{
            font-size: 1.5em; /* Increase font size */
            color: lightgreen; /* Set text color to light green */
            font-weight: bold; /* Optional: Make the text bold */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
add_bg_from_local('background.jpg')


class DataPreprocessor:
    def __init__(self):
        pass

    def replace_outliers_with_bounds(self, column):
        q1 = column.quantile(0.25)
        q3 = column.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        column = column.apply(lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x))
        return column


    def remove_outliers(self, dfp, column):
        mcolumn = dfp[column]
        q1 = mcolumn.quantile(0.25)
        q3 = mcolumn.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filtering out rows with outliers
        df_filtered = dfp[(dfp[column] >= lower_bound) & (dfp[column] <= upper_bound)]
        return df_filtered


    def fix_year(self, year):
        result = ''
        if int(year) < 25:
            result = '20'+ year
        else:
            result = '19' + year
        return result


    def preprocess(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()

        # Removing duplicated rows if any
        df.drop_duplicates(inplace = True)
        df.reset_index(drop = True, inplace = True)

        # Handling/Imputing missing values identified during data understanding using statistical values
        df.emp_length           = df.emp_length.fillna('< 1 year')
        df.revol_util           = df.revol_util.fillna(df.revol_util.mean())
        df.mort_acc             = df.mort_acc.fillna(df.mort_acc.mode()[0])
        df.pub_rec_bankruptcies = df.pub_rec_bankruptcies.fillna(df.pub_rec_bankruptcies.mode()[0])

        # Removing observations with outliers
        df = self.remove_outliers(df, 'annual_inc')
        df = self.remove_outliers(df, 'revol_bal')
        df = self.remove_outliers(df, 'revol_util')
        df = self.remove_outliers(df, 'dti')

        # Removing units in term variable and converting the values to numerical values
        df.term = df.term.str.replace(' months', '')
        df.term = df.term.apply(int)

        # Converting emp_length values to numbers
        emp_length_map = {'< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4, '5 years': 5,
                          '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10}
        df.emp_length = df.emp_length.map(emp_length_map)

        # Grouping home_ownership values to reduce unnecessary values
        df.home_ownership.replace(['NONE', 'ANY', 'OTHER'], 'RENT', inplace = True)

        # Ensure issue_d and earliest_cr_line are strings for .str operations
        df['issue_d'] = df['issue_d'].astype(str)
        df['earliest_cr_line'] = df['earliest_cr_line'].astype(str)

        # Extracting years and months from issue_d and earliest_cr_line
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        df['issue_d_year']  = df.issue_d.str[-2:].apply(self.fix_year).apply(int)
        df['cr_line_year']  = df.earliest_cr_line.str[-2:].apply(self.fix_year).apply(int)
        df['issue_d_month'] = df.issue_d.str[:3].map(month_map)
        df['cr_line_month'] = df.earliest_cr_line.str[:3].map(month_map)

        # Encoding sub_grade values to numbers
        label_encoder = LabelEncoder()
        label_encoder.fit(['A1', 'A2', 'A3', 'A4', 'A5',
                           'B1', 'B2', 'B3', 'B4', 'B5',
                           'C1', 'C2', 'C3', 'C4', 'C5',
                           'D1', 'D2', 'D3', 'D4', 'D5',
                           'E1', 'E2', 'E3', 'E4', 'E5',
                           'F1', 'F2', 'F3', 'F4', 'F5',
                           'G1', 'G2', 'G3', 'G4', 'G5'])
        df.sub_grade = label_encoder.transform(df.sub_grade)

        # Extracting zipcodes from address
        df['zipcode'] = df.address.str[-5:]

        # Dropping issue_d, earliest_cr_line, and address as key numerical information has been extracted from them
        df.drop(columns = ['issue_d', 'earliest_cr_line', 'address'], inplace = True)

        # Define the feature set based on the model training data
        expected_features = ['loan_amnt', 'term', 'int_rate', 'sub_grade', 'emp_length', 'annual_inc', 'dti', 
                             'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'mort_acc', 
                             'pub_rec_bankruptcies', 'issue_d_year', 'cr_line_year', 'issue_d_month', 'cr_line_month', 
                             'home_ownership_OWN', 'home_ownership_RENT', 'verification_status_Source Verified',
                             'verification_status_Verified', 'purpose_credit_card', 'purpose_debt_consolidation',
                             'purpose_educational', 'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
                             'purpose_medical', 'purpose_moving', 'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
                             'purpose_vacation', 'purpose_wedding', 'application_type_INDIVIDUAL', 'application_type_JOINT',
                             'zipcode_05113', 'zipcode_11650', 'zipcode_22690', 'zipcode_29597', 'zipcode_30723', 'zipcode_48052', 
                             'zipcode_70466', 'zipcode_86630', 'zipcode_93700', 'loan_status']

        # Ensure all expected features are present
        for feature in expected_features:
            if feature not in df.columns:
                df[feature] = False
        
        # Reorder columns to match the expected feature set
        df = df[expected_features]
        
        if df_input['home_ownership'].iloc[0] == "RENT":
            df.home_ownership_RENT = True
        elif df_input['home_ownership'].iloc[0] == "OWN":
            df.home_ownership_OWN = True
            
        if df_input['verification_status'].iloc[0] == "Source Verified":
            df["verification_status_Source Verified"] = True
        elif df_input['verification_status'].iloc[0] == "Verified":
            df.verification_status_Verified = True
            
        if df_input['purpose'].iloc[0] == "credit_card":
            df.purpose_credit_card = True
        elif df_input['purpose'].iloc[0] == "debt_consolidation":
            df.purpose_debt_consolidation = True
        elif df_input['purpose'].iloc[0] == "educational":
            df.purpose_educational = True
        elif df_input['purpose'].iloc[0] == "home_improvement":
            df.purpose_home_improvement = True
        elif df_input['purpose'].iloc[0] == "house":
            df.purpose_house = True
        elif df_input['purpose'].iloc[0] == "major_purchase":
            df.purpose_major_purchase = True
        elif df_input['purpose'].iloc[0] == "medical":
            df.purpose_medical = True
        elif df_input['purpose'].iloc[0] == "moving":
            df.purpose_moving = True
        elif df_input['purpose'].iloc[0] == "other":
            df.purpose_other = True
        elif df_input['purpose'].iloc[0] == "renewable_energy":
            df.purpose_renewable_energy = True
        elif df_input['purpose'].iloc[0] == "small_business":
            df.purpose_small_business = True
        elif df_input['purpose'].iloc[0] == "vacation":
            df.purpose_vacation = True
        elif df_input['purpose'].iloc[0] == "wedding":
            df.purpose_wedding = True
            
        if df_input['application_type'].iloc[0] == "INDIVIDUAL":
            df.application_type_INDIVIDUAL = True
        elif df_input['application_type'].iloc[0] == "JOINT":
            df.application_type_JOINT = True
            
        
        zipcode = df_input['address'].iloc[0][-5:]
        if zipcode == "05113":
            df.zipcode_05113 = True
        elif zipcode == "11650":
            df.zipcode_11650 = True
        elif zipcode == "22690":
            df.zipcode_22690 = True
        elif zipcode == "29597":
            df.zipcode_29597 = True
        elif zipcode == "30723":
            df.zipcode_30723 = True
        elif zipcode == "48052":
            df.zipcode_48052 = True
        elif zipcode == "70466":
            df.zipcode_70466 = True
        elif zipcode == "86630":
            df.zipcode_86630 = True
        elif zipcode == "93700":
            df.zipcode_93700 = True
            
        return df

# Load CatBoost model
model = CatBoostClassifier()
model.load_model('catboost_model.cbm')

# Streamlit app
st.markdown("<h1 style='text-align: center; color: yellow;'>Loan Status Prediction</h1>", unsafe_allow_html=True)

# Input Fields
loan_amnt = st.number_input('Loan Amount', min_value=1000, max_value=40000, value=10000, step=1000, key='loan_amnt')
term = st.selectbox('Term', ['36 months', '60 months'], index=0, key='term')
int_rate = st.number_input('Interest Rate (%)', min_value=5.0, max_value=25.0, value=10.0, step=0.1, key='int_rate')
sub_grade = st.selectbox('Sub-Grade', ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5'], index=0, key='sub_grade')
emp_length = st.selectbox('Employment Length', ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], index=0, key='emp_length')
home_ownership = st.selectbox('Home Ownership', ['RENT', 'MORTGAGE', 'OWN'], index=0, key='home_ownership')
annual_inc = st.number_input('Annual Income', min_value=10000, max_value=500000, value=50000, step=1000, key='annual_inc')
verification_status = st.selectbox('Verification Status', ['Verified', 'Source Verified', 'Not Verified'], index=0, key='verification_status')
issue_d = st.text_input('Issue Date', value='Aug-2024', key='issue_d')
purpose = st.selectbox('Purpose', ['credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'house', 'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation', 'wedding'], index=0, key='purpose')
dti = st.number_input('Debt-to-Income Ratio (%)', min_value=0.0, max_value=40.0, value=15.0, step=0.1, key='dti')
earliest_cr_line = st.text_input('Earliest Credit Line', value='Aug-2024', key='earliest_cr_line')
open_acc = st.number_input('Open Accounts', min_value=0, max_value=40, value=10, step=1, key='open_acc')
pub_rec = st.number_input('Public Records', min_value=0, max_value=10, value=0, step=1, key='pub_rec')
revol_bal = st.number_input('Revolving Balance', min_value=0, max_value=1000000, value=10000, step=100, key='revol_bal')
revol_util = st.number_input('Revolving Utilization Rate (%)', min_value=0.0, max_value=150.0, value=50.0, step=0.1, key='revol_util')
total_acc = st.number_input('Total Accounts', min_value=0, max_value=100, value=20, step=1, key='total_acc')
application_type = st.selectbox('Application Type', ['INDIVIDUAL', 'JOINT'], index=0, key='application_type')
mort_acc = st.number_input('Mortgage Accounts', min_value=0, max_value=50, value=0, step=1, key='mort_acc')
pub_rec_bankruptcies = st.number_input('Public Record Bankruptcies', min_value=0, max_value=5, value=0, step=1, key='pub_rec_bankruptcies')
address = st.text_input('Address', value='', key='address')

# Create two columns with adjusted widths for buttons
col1, col2 = st.columns([2, 1])

with col1:
    if st.button('Predict'):
        data = {
            'loan_amnt': [loan_amnt],
            'term': [term],
            'int_rate': [int_rate],
            'sub_grade': [sub_grade],
            'emp_length': [emp_length],
            'home_ownership': [home_ownership],
            'annual_inc': [annual_inc],
            'verification_status': [verification_status],
            'issue_d': [issue_d],
            'purpose': [purpose],
            'dti': [dti],
            'earliest_cr_line': [earliest_cr_line],
            'open_acc': [open_acc],
            'pub_rec': [pub_rec],
            'revol_bal': [revol_bal],
            'revol_util': [revol_util],
            'total_acc': [total_acc],
            'application_type': [application_type],
            'mort_acc': [mort_acc],
            'pub_rec_bankruptcies': [pub_rec_bankruptcies],
            'address': [address]
        }
        input_df = pd.DataFrame(data)

        # Preprocess the input data
        preprocessor = DataPreprocessor()
        processed_df = preprocessor.preprocess(input_df)

        # Make prediction
        prediction = model.predict(processed_df)
        result = 'Charged Off' if prediction[0] == 1 else 'Fully Paid'
        st.markdown(f"<h2 class='prediction-result' style='text-align: center;'>Predicted Loan Status: {result}</h2>", unsafe_allow_html=True)

with col2:
    if st.button('Reset'):
        # Reset all session state values
        for key in st.session_state.keys():
            del st.session_state[key]
        st.experimental_rerun()