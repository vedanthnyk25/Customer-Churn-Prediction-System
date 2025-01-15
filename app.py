import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Define mapping dictionaries for categorical variables
GENDER_MAP = {"Male": 0, "Female": 1}
YES_NO_MAP = {"No": 0, "Yes": 1}

# Load the model and encoders
@st.cache_resource
def load_models():
    with open("customer_churn_model.pkl", "rb") as model_file:
        model_data = pickle.load(model_file)
        return model_data["model"], model_data["feature_names"]

@st.cache_resource
def load_encoders():
    with open("encoders.pkl", "rb") as encoders_file:
        return pickle.load(encoders_file)

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def get_field_options(feature_name):
    """Return appropriate options based on field name"""
    if feature_name.lower() == 'gender':
        return list(GENDER_MAP.keys())
    elif any(word in feature_name.lower() for word in ['is_', 'has_', 'phone_service', 'multiple_lines', 'internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing']):
        return list(YES_NO_MAP.keys())
    return None

def convert_categorical_value(feature_name, value):
    """Convert human-readable categorical values to numeric"""
    if feature_name.lower() == 'gender':
        return GENDER_MAP[value]
    elif any(word in feature_name.lower() for word in ['is_', 'has_', 'phone_service', 'multiple_lines', 'internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing']):
        return YES_NO_MAP[value]
    return value

def main():
    try:
        model, feature_names = load_models()
        encoders = load_encoders()
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure model and encoder files are in the correct location.")
        return

    # Header
    st.title("🔄 Customer Churn Prediction")
    st.markdown("---")

    # Create two columns
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("📝 Customer Information")
        
        # Create form for input
        with st.form("prediction_form"):
            user_input = {}
            
            # Group numerical and categorical inputs
            numerical_inputs = {}
            categorical_inputs = {}
            
            for feature in feature_names:
                # Check if this field should have predefined options
                options = get_field_options(feature)
                
                if options is not None:
                    # For categorical fields with predefined options
                    categorical_inputs[feature] = st.selectbox(
                        f"📊 {feature.replace('_', ' ').title()}",
                        options
                    )
                elif feature in encoders:
                    # For other categorical fields
                    categorical_inputs[feature] = st.selectbox(
                        f"📊 {feature.replace('_', ' ').title()}",
                        encoders[feature].classes_
                    )
                else:
                    # For numerical fields
                    numerical_inputs[feature] = st.number_input(
                        f"📈 {feature.replace('_', ' ').title()}",
                        min_value=0.0,
                        step=0.1,
                        format="%.2f"
                    )
            
            user_input.update(numerical_inputs)
            user_input.update(categorical_inputs)
            
            submit_button = st.form_submit_button("🔍 Predict Churn")

    with col2:
        if submit_button:
            st.subheader("🎯 Prediction Results")
            
            # Show loading spinner during prediction
            with st.spinner("Analyzing customer data..."):
                # Prepare input for prediction
                input_data = {}
                for feature, value in user_input.items():
                    # Convert categorical values if needed
                    input_data[feature] = convert_categorical_value(feature, value)
                
                input_df = pd.DataFrame([input_data])
                
                # Encode remaining categorical features that use encoders
                for col, encoder in encoders.items():
                    if col in input_df and not any(word in col.lower() for word in ['gender', 'is_', 'has_', 'phone_service', 'multiple_lines', 'internet_service', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing']):
                        input_df[col] = encoder.transform(input_df[col])
                
                # Make prediction
                probabilities = model.predict_proba(input_df)[0]
                churn_probability = probabilities[1]
                prediction = "High Risk of Churn" if churn_probability > 0.5 else "Low Risk of Churn"
                
                # Display gauge chart
                st.plotly_chart(create_gauge_chart(churn_probability))
                
                # Display prediction results
                result_color = "salmon" if churn_probability > 0.5 else "lightgreen"
                st.markdown(f"""
                    <div style="background-color: {result_color}; padding: 20px; border-radius: 10px;">
                        <h3 style="color: black;">Prediction: {prediction}</h3>
                        <p style="color: black; font-size: 18px;">
                            Churn Probability: {churn_probability:.1%}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add timestamp
                st.caption(f"Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Recommendations based on prediction
                st.subheader("📋 Recommendations")
                if churn_probability > 0.5:
                    st.warning("""
                        - Immediate customer engagement recommended
                        - Review pricing and service plans
                        - Schedule customer satisfaction survey
                        - Consider offering retention incentives
                    """)
                else:
                    st.success("""
                        - Continue monitoring customer satisfaction
                        - Consider upselling opportunities
                        - Maintain regular communication
                        - Collect feedback for service improvement
                    """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center">
            <p>Built with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()