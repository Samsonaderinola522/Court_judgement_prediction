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
    
#Create lists of sample values for each input field:
first_party_samples = ['Apple Inc.', 'Google LLC', 'Microsoft Corporation']
    
second_party_samples = ['Samsung Electronics Co., Ltd.', 'Huawei Technologies Co., Ltd.', 'Nokia Corporation']
    
issue_area_samples = ["Civil Rights", "Due Process", "First Amendment", "Criminal Procedure", "Privacy", "Federal Taxation", "Economic Activity", "Judicial Power", "Federalism", "Attorneys", "Miscellaneous", "Interstate Relations", "Private Action", "Others"]
    
case_facts_samples = [
        'This is a case about patent infringement of smart phone technologies.',
        'This is a contract dispute involving a major software services provider and a global bank.',
        'This is a constitutional law case involving freedom of speech and expression.'
    ]


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

# Define your Streamlit app
def main():
    # Set the title of your app
    st.title('Judgment Prediction App')
    
    # Displaying the image with caption
    st.image("fuoye-logo.png")
    
    # Add text input fields for users to enter parties involved
    first_party_input = st.text_input('Enter first party:', value=first_party_inputx)
    second_party_input = st.text_input('Enter second party:', value=second_party_inputx)

    # Add a text input field for issue area
    options = ["Civil Rights", "Due Process", "First Amendment", "Criminal Procedure", "Privacy", "Federal Taxation", "Economic Activity", "Judicial Power", "Federalism", "Attorneys", "Miscellaneous", "Interstate Relations", "Private Action", "Others"]
    issue_area = st.selectbox("Select issue area:", options, value=issue_areax)

    # Add a text input field for users to enter their case facts
    case_facts_input = st.text_area('Enter case facts:', value=case_facts_inputx)
    
    # Add a button to make predictions
    if st.button('Predict Judgment'):
        # Make a prediction and display the result
        prediction = predict_partystub(case_facts_input)
        if prediction == 'First Party':
            #st.write(f'The judgment is likely to favor {first_party_input}.')
            st.markdown(f"<h1 style='text-align:center; color: green;'> The judgment is likely to favor. </h1>", unsafe_allow_html=True)
        else:
            #st.write(f'The judgment is likely to favor {second_party_input}.')
            st.markdown(f"<h1 style='text-align:center; color: green;'> The judgment is likely to favor. </h1>", unsafe_allow_html=True)
    # Add a button for auto filling out form
    if st.button("Autofill-Sample"):
       # Randomly select one value from each list
       first_party_inputx = random.choice(first_party_samples)
       second_party_inputx = random.choice(second_party_samples)
       issue_areax = random.choice(issue_area_samples)
       case_facts_inputx = random.choice(case_facts_samples)
    
if __name__ == '__main__':
    main()


#This code loads Keras LSTM model and tokenizer, defines a function to make predictions based on case facts, and defines your Streamlit app with two sections: "Case Information" and "Prediction". You can customize this code to fit your specific use case.
