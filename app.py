# Import required libraries
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import time as t

# Define a function to predict the energy consumption from user inputs
def make_prediction(X_input):
    
    # Load the saved model weights
    saved_coef = joblib.load("reg_coef.pkl")
    saved_intercept = joblib.load("reg_intercept.pkl")
    loaded_model = LinearRegression()
    loaded_model.coef_ = saved_coef
    loaded_model.intercept_ = saved_intercept
    pred_value = loaded_model.predict(X_input.values)
    return (pred_value)

#################################################################################################################
# Define page title 
st.set_page_config(page_title="Energy Consumption Calculator", layout='wide')
#################################################################################################################

# Heading
st.title("Electricity Consumption Analysis and Prediction")
# Area in sqft 
area = st.number_input("Gross area of apartment in sqft", min_value=300, max_value=4000,value=300)
#st.write(area)

# No of bedrooms
bedrooms = st.number_input("No of bedrooms", min_value=1, max_value=5, value=1)
#st.write(bedrooms)

# Occupancy
occupancy = st.number_input("Occupancy", min_value=1, max_value=15,value=1)
#st.write(occupancy)

# AC units 
no_of_ac = st.number_input("No of ac units", min_value=1, max_value=10,value=1)
#st.write(no_of_ac)

# HV appliances 
no_of_hvac = st.number_input("No of HV appliances(water heaters, refridgerator, washing machine, dishwasher etc)", min_value=1, max_value=15,value=1)
#st.write(no_of_hvac)

# Lighting units 
no_of_lights = st.number_input("No of lighting units", min_value=1, max_value=25,value=1)
#st.write(no_of_lights)

# Avg thermostat reading
inside_temp = st.slider("AC temperature in celcius", min_value=15.0, max_value=30.0, value=25.0, step=0.5)
#st.write(inside_temp)

# Outside temperature
outside_temp = st.slider("Outside temperature in celcius", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
#st.write(outside_temp)


# Create a dictionary with the variables and their values
data = {
    "Area": [area],
    "Bedroooms": [bedrooms],
    "occupancy": [occupancy],
    "no_of_ac": [no_of_ac],
    "no_of_hvac": [no_of_hvac],
    "no_of_lights":[no_of_lights],
    "inside_temp":[inside_temp],
    "outside_temp":[outside_temp]
}

# Create a DataFrame from the dictionary
data_pred = pd.DataFrame(data)
X_pred = data_pred[["Area","Bedroooms","occupancy","no_of_ac","no_of_hvac","no_of_lights","inside_temp","outside_temp"]]

# Predict the energy consumption from user input
prediction = make_prediction(X_pred)
#print(prediction)
# Submit button
if st.button("Submit"):
    with st.spinner('Predicting..'):
        t.sleep(3)
    # Display the predicted value
    st.header("Predicted Consumption in kWh")
    st.header(np.round(prediction[0],2))
    # Create data for the pie chart
    hvac_usg = (no_of_hvac+no_of_ac)*20
    light_usg = (no_of_lights)*1
    miscellaneous_usg = .05*hvac_usg
    sizes = [hvac_usg, light_usg,miscellaneous_usg]  # Sizes/percentages for each category
    palette_color = sns.color_palette('dark') 
    labels = 'HVAC', 'Lighting', 'Miscellaneous'
    # Create a pie chart
    plt.figure(figsize=(15,5))
    plt.style.use('dark_background')
    plt.pie(sizes, labels=labels, colors=palette_color,textprops={'fontsize': 8})
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.title("Electricity consumption distribution",fontsize=14)

    # Display the pie chart in the Streamlit app
    plt.show()
    st.pyplot(plt, clear_figure=None, use_container_width=True)
# # Display a plot
# import matplotlib.pyplot as plt
# st.pyplot(plt)

# # Display data in a table
# import pandas as pd
# data = pd.DataFrame({'Column 1': [1, 2, 3], 'Column 2': [4, 5, 6]})
# st.write(data)
