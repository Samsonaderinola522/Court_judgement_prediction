
import random
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Load the model
model = load_model('my_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    #tokenizer = tokenizer_from_json(data)

# A fuctio for tokeizing ad padding
def tokenize_and_pad_input(input_text, tokenizer, max_len):
    # Tokenize the text
    tokens = tokenizer.texts_to_sequences([input_text])
    
    # Pad sequences
    padded_tokens = pad_sequences(tokens, maxlen=max_len)
    
    return padded_tokens

# A function to predict party -2
def predict_party2(input_text, model=model, tokenizer=tokenizer, max_len=407):
    # Tokenize and pad the input
    padded_tokens = tokenize_and_pad_input(input_text, tokenizer, max_len)
    
    # Get binary prediction probabilities for the input data
    prediction_probabilities = model.predict(padded_tokens)[0]
    
    # Determine the class based on prediction probability threshold (e.g., 0.5)
    if prediction_probabilities > 0.5:
        return "First party"
    else:
        return "Second party"

# STU A function to predict party 
def predict_partystub(input_text):
    
    prediction_probabilities = 3
    # Determine the class based on prediction probability threshold (e.g., 0.5)
    if prediction_probabilities > 0.5:
        return "First party"
    else:
        return "Second party"        

# Define a function to make predictions
def predict_party(case_facts):
    # Tokenize the input text
    case_facts = tokenizer.texts_to_sequences([case_facts])
    # Pad the input text
    case_facts = pad_sequences(case_facts, maxlen=MAX_SEQUENCE_LENGTH)
    # Make a prediction
    prediction = model.predict(case_facts)[0][0]
    if prediction > 0.5:
        return 'First Party'
    else:
        return 'Second Party'
        
def empty_form():
    # Add text input fields for users to enter parties involved
    first_party_input = st.text_input('Enter first party:')
    second_party_input = st.text_input('Enter second party:')
    
    # Add a text input field for issue area
    options = ["Civil Rights", "Due Process", "First Amendment", "Criminal Procedure", "Privacy", "Federal Taxation", "Economic Activity", "Judicial Power", "Federalism", "Attorneys", "Miscellaneous", "Interstate Relations", "Private Action", "Others"]
    issue_area = st.selectbox("Select issue area:", options)
    
    # Add a text input field for users to enter their case facts
    case_facts_input = st.text_area('Enter case facts:')
    
    # Add a button to make predictions
    if st.button('Predict Judgment'):
        # Make a prediction and display the result
        prediction = predict_partystub(case_facts_input)
        if prediction == 'First Party':
            #st.write(f'The judgment is likely to favor {first_party_input}.')
            st.markdown(f"<h1 style='text-align:center; color: green;'> The judgment is likely to favor {first_party_input}. </h1>", unsafe_allow_html=True)
        else:
            #st.write(f'The judgment is likely to favor {second_party_input}.')
            st.markdown(f"<h1 style='text-align:center; color: green;'> The judgment is likely to favor {second_party_input}. </h1>", unsafe_allow_html=True)
    
def prefilled_form():
    # Define sample input values for each input field
    first_party_samples = ["John Doe", "Jane Smith", "XYZ Corporation"]
    second_party_samples = ["Jack Johnson", "Acme Corporation", "Mary Lee"]
    issue_area_samples = [0, 1, 2, 3, 4, 
                          5, 6, 7, 8, 
                          9, 10, 11, 12, 
                          13]
    case_facts_samples = ["This is the first party's statement. The second party disagrees...", 
                          "The parties are in dispute over a contract governing...", 
                          "This case involves an alleged violation of privacy..."]
    case = random.randint(0,3)
    # Randomly select a value from each sample list
    first_party_value = first_party_samples[case]
    second_party_value = second_party_samples[case]
    issue_area_value = issue_area_samples[case]
    case_facts_value = case_facts_samples[case]
    
    # Add text input fields for users to enter parties involved
    first_party_input = st.text_input('Enter first party:', value=first_party_value)
    second_party_input = st.text_input('Enter second party:', value=second_party_value)
    
    # Add a text input field for issue area
    options = ["Civil Rights", "Due Process", "First Amendment", "Criminal Procedure", "Privacy", "Federal Taxation", "Economic Activity", "Judicial Power", "Federalism", "Attorneys", "Miscellaneous", "Interstate Relations", "Private Action", "Others"]
    issue_area = st.selectbox("Select issue area:", options, index=issue_area_value)
    
    # Add a text input field for users to enter their case facts
    case_facts_input = st.text_area('Enter case facts:', value=case_facts_value)
    
    # Add a button to make predictions
    if st.button('Predict Judgment'):
        # Make a prediction and display the result
        prediction = predict_partystub(case_facts_input)
        if prediction == 'First Party':
            #st.write(f'The judgment is likely to favor {first_party_input}.')
            st.markdown(f"<h1 style='text-align:center; color: green;'> The judgment is likely to favor {first_party_input}. </h1>", unsafe_allow_html=True)
        else:
            #st.write(f'The judgment is likely to favor {second_party_input}.')
            st.markdown(f"<h1 style='text-align:center; color: green;'> The judgment is likely to favor {second_party_input}. </h1>", unsafe_allow_html=True)
    
st.session_state.form_mode = None  
    
# Define your Streamlit app
def main():
    # Displaying the image with caption
    st.image("fuoye-logo.png")
    
    # Set the title of your app
    st.title('Judgment Prediction App')
    
    if st.session_state.form_mode is None:
        # Display initial form mode selector
        form_mode = st.radio("Select Form Mode", ["Empty Form", "Prefilled Form"])
        #if st.button("Submit"):
        st.session_state.form_mode = form_mode
            #st.write(form_mode)
            #st.write(st.session_state.form_mode)
    
    if st.session_state.form_mode == "Empty Form":
        empty_form()
        #st.write(form_mode)
         
    if st.session_state.form_mode == "Prefilled Form":
        #st.write(form_mode)
        prefilled_form()
 
       
if __name__ == '__main__':
    main()


#This code loads Keras LSTM model and tokenizer, defines a function to make predictions based on case facts, and defines your Streamlit app with two sections: "Case Information" and "Prediction". You can customize this code to fit your specific use case.
