import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import random

# Load the model
model = load_model('my_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    #tokenizer = tokenizer_from_json(data)

# Set the max sequence length for tokenization/padding
MAX_SEQUENCE_LENGTH = 407

# A function for tokenizing and padding inputs
def tokenize_and_pad_input(input_text, tokenizer, max_len):
    # Tokenize the text
    tokens = tokenizer.texts_to_sequences([input_text])
    
    # Pad sequences
    padded_tokens = pad_sequences(tokens, maxlen=max_len)
    
    return padded_tokens

# A function to predict party 
def predict_party(input_text, model=model, tokenizer=tokenizer, max_len=MAX_SEQUENCE_LENGTH):
    # Tokenize and pad the input
    padded_tokens = tokenize_and_pad_input(input_text, tokenizer, max_len)
    
    # Get binary prediction probabilities for the input data
    prediction_probabilities = model.predict(padded_tokens)[0]
    
    # Determine the class based on prediction probability threshold (e.g., 0.5)
    if prediction_probabilities > 0.5:
        return "First party"
    else:
        return "Second party"

# Define your Streamlit app
def main():
    # Set the title of your app
    st.title('Judgment Prediction App')
    
    # Displaying the image with caption
    st.image("fuoye-logo.png")
    
    # Define sample input values for each input field
    first_party_samples = ["John Doe", "Jane Smith", "XYZ Corporation"]
    second_party_samples = ["Jack Johnson", "Acme Corporation", "Mary Lee"]
    issue_area_samples = ["Civil Rights", "Due Process", "First Amendment", "Criminal Procedure", "Privacy", 
                          "Federal Taxation", "Economic Activity", "Judicial Power", "Federalism", 
                          "Attorneys", "Miscellaneous", "Interstate Relations", "Private Action", 
                          "Others"]
    case_facts_samples = ["This is the first party's statement. The second party disagrees...", 
                          "The parties are in dispute over a contract governing...", 
                          "This case involves an alleged violation of privacy..."]
    
    # Add text input fields for users to enter parties involved
    first_party_input = st.text_input('Enter first party:')
    second_party_input = st.text_input('Enter second party:')

    # Add a text input field for issue area
    options = ["Civil Rights", "Due Process", "First Amendment", "Criminal Procedure", "Privacy", "Federal Taxation", "Economic Activity", "Judicial Power", "Federalism", "Attorneys", "Miscellaneous", "Interstate Relations", "Private Action", "Others"]
    issue_area = st.selectbox("Select issue area:", options)

    # Add a text input field for users to enter their case facts
    case_facts_input = st.text_area('Enter case facts:')
    
    # Add an autofill button that fills the input fields with sample values
    if st.button('Autofill-Sample'):
        # Randomly select a value from each sample list
        first_party_value = random.choice(first_party_samples)
        second_party_value = random.choice(second_party_samples)
        issue_area_value = random.choice(issue_area_samples)
        case_facts_value = random.choice(case_facts_samples)
        
        # Update the input fields with the selected values
        first_party_input = st.text_input('Enter first party:', value=first_party_value)
        second_party_input = st.text_input('Enter second party:', value=second_party_value)
        issue_area = st.selectbox("Select issue area:", options, index=options.index(issue_area_value))
        case_facts_input = st.text_area('Enter case facts:', value=case_facts_value)
    
    # Add a button to make predictions
    if st.button('Predict Judgment'):
        # Make a prediction and display the result
        prediction = predict_party(case_facts_input)
        if prediction == 'First party':
            st.write(f'The judgment is likely to favor {first_party_input}.')
        else:
            st.write(f'The judgment is likely to favor {second_party_input}.')

if __name__ == '__main__':
    main()