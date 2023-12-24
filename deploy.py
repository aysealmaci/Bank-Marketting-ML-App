import streamlit as st
import joblib
import pandas as pd

model = joblib.load(open(r'C:\Users\Melisa\Desktop\ADA 442\Project\bank-additional.sav', 'rb'))


def prediction(features):
    # Mapping for label encoding
    label_mapping = {
        'job': {'admin': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3, 'management': 4,
                'retired': 5, 'self-employed': 6, 'services': 7, 'student': 8, 'technician': 9, 'unemployed': 10},
        'marital': {'divorced': 0, 'married': 1, 'single': 2},
        'education': {'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3, 'illiterate': 4,
                      'professional.course': 5, 'university.degree': 6},
        'housing': {'no': 0, 'yes': 1},
        'loan': {'no': 0, 'yes': 1},
        'contact': {'cellular': 0, 'telephone': 1},
        'poutcome': {'failure': 0, 'nonexistent': 1, 'success': 2}
    }

    # Apply label encoding for categorical columns
    for column, mapping in label_mapping.items():
        features[column] = features[column].map(mapping)

    # Make the prediction
    prediction = model.predict(features)
    return prediction[0]


def main():
    # Set the background color using custom CSS
    background_color = """
        <style>
            body {
                background-color: #fffcce;
                margin: 0;
                padding: 0;
                font-family: 'Arial', sans-serif;
            }
        </style>
    """

    st.markdown(background_color, unsafe_allow_html=True)

    # Frontend elements of the web page with ornamentation
    html_temp = """ 
        <div style="
            padding: 20px; 
            border-radius: 15px; 
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            background: linear-gradient(to right, #fffcce, #edc16a); /* Use your desired color stops */
        "> 
            <h1 style="color:black; text-align:center; font-size: 2.5rem; letter-spacing: 2px;">Bank Marketing Machine Learning App</h1> 
        </div> 
    """
    full = background_color + html_temp
    # Display the frontend aspect
    st.markdown(full, unsafe_allow_html=True)

    # Add some instructions or description
    st.write("""
    ### Instructions:
    1. Enter the required information in the input fields.
    2. Click the "Predict" button to see the prediction result.
    3. Explore the results and enjoy using the app!

    """)

    # following lines create boxes in which user can enter data required to make prediction
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    job = st.selectbox('Job', (
        'admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services',
        'student', 'technician', 'unemployed'))
    marital = st.selectbox('Marital', ('divorced', 'married', 'single'))
    education = st.selectbox('Education', (
        'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree'))
    housing = st.selectbox('Housing', ('no', 'yes'))
    loan = st.selectbox('Loan', ('no', 'yes'))
    contact = st.selectbox('Contact', ('cellular', 'telephone'))
    duration = st.number_input("Duration", step=1)
    campaign = st.number_input("Campaign", step=1)
    pdays = st.number_input("PDays", step=1)
    previous = st.number_input("Previous", step=1)
    poutcome = st.selectbox('Poutcome', ('failure', 'nonexistent', 'success'))
    emp_var_rate = st.number_input("Employment Variation Rate")
    cons_price_idx = st.number_input("Consumer Price Index")
    cons_conf_idx = st.number_input("Consumer Confidence Index")
    # euribor3m = st.number_input("Euribor 3 month rate")
    nr_employed = st.number_input("Number of Employees")

    # when 'Predict' is clicked, make the prediction and store it
    if st.button("Predict"):
        features = pd.DataFrame({
            'age': [age],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'nr.employed': [nr_employed],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'duration': [duration],
            'poutcome': [poutcome]})

        prediction_result = prediction(features)
        # Display the prediction result

        # Frontend elements for the prediction result
        st.subheader("Prediction Result:")
        if prediction_result == 1:
            st.success('Yes, the customer is likely to subscribe.')
        else:
            st.error('No, the customer is unlikely to subscribe.')

        # Add some styling to the prediction result
        st.markdown(
            """
            <style>
                .stSubheader {
                    color: #2c3e50; /* Subheader text color */
                    font-size: 1.5rem; /* Subheader font size */
                    margin-top: 20px; /* Top margin */
                }
                .stSuccess, .stError {
                    background-color: #d4edda; /* Background color for success and error messages */
                    color: #155724; /* Text color for success and error messages */
                    padding: 10px; /* Padding for success and error messages */
                    border-radius: 5px; /* Border radius for success and error messages */
                    margin-top: 10px; /* Top margin for success and error messages */
                }
            </style>
            """,
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    main()

    st.markdown(
        """
         Developed by:
         - Ayşe Almacı
         - Melisa Yüncü
         - Neslihan Pelin Metin
         - Rabiye Nur Balcı
         """
    )