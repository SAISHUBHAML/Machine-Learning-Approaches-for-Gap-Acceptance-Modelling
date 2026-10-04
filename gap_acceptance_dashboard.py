import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import time
import re 

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Gap Acceptance Predictor & Explainer")

# --- Load Saved Objects ---
pipeline = None
processed_feature_names = None
shap_background_data = None 
input_options = None
model_loaded = False
X_train_cols = None # To store original feature order used for input_df

# Cache loading to speed up reruns
@st.cache_resource
def load_resources():
    # Loads pipeline, feature names, background data, input options, and original column order
    try:
        _pipeline = joblib.load('catboost_pipeline.joblib') 
        _processed_feature_names = joblib.load('processed_feature_names.joblib') 
        _shap_background_data = pd.read_pickle('shap_background_data.pkl') 
        _input_options = joblib.load('input_options.joblib') 
        
        try:
             _X_train_cols = joblib.load('X_train_cols.joblib') 
             print("Loaded original X_train column order.")
        except FileNotFoundError:
             print("Warning: 'X_train_cols.joblib' not found. Using default order for input.")
             _cat_cols = ['Sub_Vehicle', 'Through_Veh', 'LagGap']
             _num_cols = ['Speed_kmhr', 'Spatial_Gap', 'Time_Gap']
             _X_train_cols = _cat_cols + _num_cols 
             print(f"Using assumed original feature order: {_X_train_cols}")
        except Exception as e_cols:
             print(f"Error loading X_train columns: {e_cols}. Using default order.")
             _cat_cols = ['Sub_Vehicle', 'Through_Veh', 'LagGap']
             _num_cols = ['Speed_kmhr', 'Spatial_Gap', 'Time_Gap']
             _X_train_cols = _cat_cols + _num_cols 
             print(f"Using assumed original feature order: {_X_train_cols}")

        print("Successfully loaded model pipeline, feature names, background data, and input options.")
        if isinstance(_processed_feature_names, np.ndarray):
             _processed_feature_names = _processed_feature_names.tolist()
             
        return _pipeline, _processed_feature_names, _shap_background_data, _input_options, True, _X_train_cols
        
    except FileNotFoundError:
        st.error("Error: One or more required files missing. Please ensure they are present and generated from the latest model training run (with interaction features).")
        return None, None, None, None, False, None
    except Exception as e:
        st.error(f"An error occurred loading files: {e}")
        return None, None, None, None, False, None

pipeline, processed_feature_names, shap_background_data, input_options, model_loaded, X_train_cols = load_resources()

# --- Initialize SHAP Explainer ---
@st.cache_resource
def get_shap_explainer(_pipeline, _shap_background_data, _processed_feature_names):
    # Initializes SHAP TreeExplainer using loaded components
    if _pipeline and _shap_background_data is not None and _processed_feature_names is not None:
        try:
            catboost_model = _pipeline.named_steps['classifier']
            if list(_shap_background_data.columns) == list(_processed_feature_names):
                 _explainer = shap.TreeExplainer(catboost_model, data=_shap_background_data)
                 print("SHAP TreeExplainer initialized successfully with background data.")
                 return _explainer
            else:
                 st.warning("SHAP background data columns mismatch. Initializing explainer without background data.")
                 _explainer = shap.TreeExplainer(catboost_model) 
                 print("Initialized SHAP Explainer without background data.")
                 return _explainer
        except Exception as e:
            st.error(f"Error initializing SHAP explainer: {e}")
            return None
    return None

explainer = get_shap_explainer(pipeline, shap_background_data, processed_feature_names)

# --- Calculate ONLY Expected Value from Background Data ---
@st.cache_data 
def calculate_expected_value(_explainer):
    # Calculates expected value 
    if _explainer is None: return None
    print("Calculating SHAP expected value...")
    start_time = time.time()
    _expected_value = _explainer.expected_value
    print(f"  (Took {time.time() - start_time:.2f} seconds)")
    _expected_value_pos = _expected_value[1] if isinstance(_expected_value, list) and len(_expected_value) == 2 else _expected_value
    return _expected_value_pos

expected_value_to_plot = calculate_expected_value(explainer) 


# --- Page Title ---
st.title("🚗 Gap Acceptance Predictor & Explainer")
st.markdown("Select scenario features below to predict driver behavior and see which factors influenced that prediction.")
st.markdown("---")

# --- Define Feature Lists (Matching Cell 5 in Colab used for training) ---
# These MUST match the lists used when defining the 'preprocessor' 
numerical_cols = [
   'Speed_kmhr', 'Spatial_Gap', 'Time_Gap', 
   'FE_TimeGap_x_Speed', 'FE_SpatialGap_x_Speed'
]
categorical_cols = [
   'Sub_Vehicle', 'Through_Veh', 'LagGap',
   'FE_Sub_x_Through', 'FE_Sub_x_LagGap'  
]
# This is the final order of columns the preprocessor expects
final_feature_order = numerical_cols + categorical_cols 

# These are the columns the user needs to input
orig_cat_cols = ['Sub_Vehicle', 'Through_Veh', 'LagGap'] 
orig_num_cols = ['Speed_kmhr', 'Spatial_Gap', 'Time_Gap'] 
original_input_feature_order = X_train_cols if X_train_cols else orig_cat_cols + orig_num_cols

# --- Sidebar for Inputs ---
input_data = {} 
if model_loaded and input_options:
    st.sidebar.header("Input Scenario Features:")
    
    # --- Categorical Inputs ---
    st.sidebar.subheader("Categorical Factors") 
    for col in orig_cat_cols:
        # ... (selectbox code remains the same) ...
        if col in input_options and isinstance(input_options[col], list): 
            options = sorted(input_options[col])
            default_option = options[0] if options else None 
            default_index = options.index(default_option) if default_option in options else 0
            input_data[col] = st.sidebar.selectbox(f"Select {col.replace('_', ' ')}:", options, index=default_index)
        else:
            st.sidebar.warning(f"Options for '{col}' not found or invalid.")
            input_data[col] = None

    # --- Numerical Inputs ---
    st.sidebar.subheader("Numerical Factors")
    for col in orig_num_cols:
        # ... (slider code remains the same) ...
         if col in input_options and isinstance(input_options[col], dict):
             min_val = input_options[col].get('min', 0.0)
             max_val = input_options[col].get('max', 100.0)
             median_val = input_options[col].get('median', (min_val + max_val) / 2)
             step = max(0.1, (max_val - min_val) / 100) if max_val > min_val else 0.1
             value_val = max(min_val, min(max_val, median_val))
             input_data[col] = st.sidebar.slider(
                 f"{col.replace('_', ' ')}:", 
                 min_value=float(min_val), max_value=float(max_val), 
                 value=float(value_val), step=float(step), format="%.1f"
             )
         else:
             st.sidebar.warning(f"Range for '{col}' not found or invalid.")
             input_data[col] = None

    # --- Create DataFrame from User Inputs AND Apply Feature Engineering ---
    input_failed = False
    for col, val in input_data.items():
        if val is None:
            st.error(f"Input for '{col}' failed to load or was not set. Cannot proceed.")
            input_failed = True
            
    if input_failed:
        st.stop() 

    try:
        # Create DataFrame with only the user-input columns first
        input_df_orig = pd.DataFrame([input_data])[original_input_feature_order] 
        
        # *** Apply Feature Engineering to the input DataFrame ***
        input_df_fe = input_df_orig.copy()
        # Interaction 1: Time Gap x Speed
        if 'Time_Gap' in input_df_fe and 'Speed_kmhr' in input_df_fe:
            input_df_fe['FE_TimeGap_x_Speed'] = input_df_fe['Time_Gap'] * input_df_fe['Speed_kmhr']
        # Interaction 2: Spatial Gap x Speed
        if 'Spatial_Gap' in input_df_fe and 'Speed_kmhr' in input_df_fe:
            input_df_fe['FE_SpatialGap_x_Speed'] = input_df_fe['Spatial_Gap'] * input_df_fe['Speed_kmhr']
        # Interaction 3: Sub x Through (Combined Categorical)
        if 'Sub_Vehicle' in input_df_fe and 'Through_Veh' in input_df_fe:
            input_df_fe['FE_Sub_x_Through'] = input_df_fe['Sub_Vehicle'].astype(str) + '_' + input_df_fe['Through_Veh'].astype(str)
        # Interaction 4: Sub x LagGap (Combined Categorical)
        if 'Sub_Vehicle' in input_df_fe and 'LagGap' in input_df_fe:
            input_df_fe['FE_Sub_x_LagGap'] = input_df_fe['Sub_Vehicle'].astype(str) + '_' + input_df_fe['LagGap'].astype(str)
        
        # Ensure correct final column order before displaying
        missing_fe_cols = [col for col in final_feature_order if col not in input_df_fe.columns]
        if missing_fe_cols:
             st.error(f"Engineered features missing from input after FE step: {missing_fe_cols}")
             st.stop()
        input_df_to_display = input_df_fe[final_feature_order] # Use this for display & prediction

        # *** Display the DataFrame WITH engineered features ***
        st.subheader("Selected Scenario (Input + Engineered Features):")
        st.dataframe(input_df_to_display, use_container_width=True) 
        
    except KeyError as e:
        st.error(f"Error creating/displaying input DataFrame. Column mismatch likely: {e}. Check feature lists.")
        st.stop()
    except Exception as e:
        st.error(f"Error creating/displaying input DataFrame: {e}")
        st.stop()

    # --- Prediction and Explanation Button ---
    if st.button("Predict & Explain"):
        if pipeline is None or explainer is None:
             st.error("Model pipeline or SHAP explainer not loaded/initialized. Cannot proceed.")
        else:
            with st.spinner("Processing Prediction & Explanation..."): 
                try:
                    st.markdown("---")
                    st.subheader("📈 Prediction & Explanation")
                    
                    # --- 1. Preprocess the DataFrame that ALREADY has FE features ---
                    # Use input_df_to_display which has the correct columns and order
                    input_processed = pipeline.named_steps['preprocessor'].transform(input_df_to_display) 
                    if processed_feature_names is None:
                         st.error("Processed feature names not loaded. Cannot proceed with SHAP.")
                         st.stop()
                    input_processed_df = pd.DataFrame(input_processed, columns=processed_feature_names, index=input_df_to_display.index)

                    # --- 2. Make Prediction ---
                    # ... (Prediction code remains the same) ...
                    prediction_proba = pipeline.named_steps['classifier'].predict_proba(input_processed_df)[0] 
                    prediction = pipeline.named_steps['classifier'].predict(input_processed_df)[0] 
                    pred_class = "Accept" if prediction == 1 else "Reject"
                    pred_confidence = prediction_proba[prediction] 

                    col1, col2 = st.columns(2) 
                    with col1:
                        st.metric(label="Model Prediction", value=pred_class)
                    with col2:
                        st.metric(label="Confidence (Probability)", value=f"{pred_confidence:.1%}")

                    # --- 3. Generate SHAP Bar Plot for Local Explanation ---
                    # ... (Bar Plot code remains the same) ...
                    st.markdown("**Local Explanation (Feature Contributions to this Prediction):**")
                    st.markdown("*(Shows magnitude and direction of each feature's impact)*")
                    
                    try:
                        shap_values_instance = explainer.shap_values(input_processed_df.iloc[[0]]) 
                        
                        if isinstance(shap_values_instance, list) and len(shap_values_instance) == 2:
                            shap_values_to_plot = shap_values_instance[1][0] 
                        else: 
                            shap_values_to_plot = shap_values_instance[0] if isinstance(shap_values_instance, list) else shap_values_instance[0] 

                        fig_bar, ax_bar = plt.subplots()
                        shap.bar_plot(
                            shap_values_to_plot, 
                            features=input_processed_df.iloc[0].values, 
                            feature_names=processed_feature_names, 
                            max_display=15, 
                            show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig_bar, bbox_inches='tight', dpi=150) 
                        plt.close(fig_bar) 

                    except Exception as plot_e:
                         st.error(f"Error generating SHAP bar plot: {plot_e}")
                         st.markdown("_(Bar plot generation failed.)_")
                         import traceback
                         st.text(traceback.format_exc()) 
                         
                except Exception as e:
                    st.error(f"An error occurred during prediction/explanation: {e}")
                    import traceback
                    st.text(traceback.format_exc()) 

else: # Handles case where model/options didn't load
    st.warning("Model components not loaded successfully. Please ensure the required joblib/pickle files are in the same directory as the script and restart.")

# --- Footer ---
st.markdown("---")
st.caption("Gap Acceptance Behavior Model Dashboard - IIT Hyderabad")

