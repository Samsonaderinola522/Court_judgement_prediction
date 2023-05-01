#may updates

import random
import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences
MAX_SEQUENCE_LENGTH = 407

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
    
@st.cache_data    
def prefilled_form(unused_x):
    # Define sample input values for each input field
    first_party_samples = ["Jane Roe", "Peter Stanley, Sr. ", "John Giglio", "Sally Reed", "Marvin Miller", ""]
    second_party_samples = ["Henry Wade", "Illinois", "United States", "Cecil Reed", "California", ""]
    issue_area_samples = [0, 0, 1, 0, 2, 
                         ]
    case_facts_samples = ["In 1970, Jane Roe (a fictional name used in court documents to protect the plaintiff’s identity) filed a lawsuit against Henry Wade, the district attorney of Dallas County, Texas, where she resided, challenging a Texas law making abortion illegal except by a doctor’s orders to save a woman’s life. In her lawsuit, Roe alleged that the state laws were unconstitutionally vague and abridged her right of personal privacy, protected by the First, Fourth, Fifth, Ninth, and Fourteenth Amendments.", 
                          "Joan Stanley had three children with Peter Stanley.  The Stanleys never married, but lived together off and on for 18 years.  When Joan died, the State of Illinois took the children.  Under Illinois law, unwed fathers were presumed unfit parents regardless of their actual fitness and their children became wards of the state.  Peter appealed the decision, arguing that the Illinois law violated the Equal Protection Clause of the Fourteenth Amendment because unwed mothers were not deprived of their children without a showing that they were actually unfit parents.  The Illinois Supreme Court rejected Stanley’s Equal Protection claim, holding that his actual fitness as a parent was irrelevant because he and the children’s mother were unmarried.", 
                          "John Giglio was convicted of passing forged money orders.  While his appeal to the U.S. Court of Appeals for the Second Circuit was pending, Giglio’s counsel discovered new evidence. The evidence indicated that the prosecution failed to disclose that it promised a key witness immunity from prosecution in exchange for testimony against Giglio.  The district court denied Giglio’s motion for a new trial, finding that the error did not affect the verdict.  The Court of Appeals affirmed.",
                          "The Idaho Probate Code specified that 'males must be preferred to females' in appointing administrators of estates. After the death of their adopted son, both Sally and Cecil Reed sought to be named the administrator of their son's estate (the Reeds were separated). According to the Probate Code, Cecil was appointed administrator and Sally challenged the law in court. ",
                          "Miller, after conducting a mass mailing campaign to advertise the sale of 'adult' material, was convicted of violating a California statute prohibiting the distribution of obscene material. Some unwilling recipients of Miller's brochures complained to the police, initiating the legal proceedings. "
                         ]
    case = random.randint(0,4)
    # Randomly select a value from each sample list
    first_party_value = first_party_samples[case]
    second_party_value = second_party_samples[case]
    issue_area_value = issue_area_samples[case]
    case_facts_value = case_facts_samples[case]
    
    return first_party_value, second_party_value, issue_area_value, case_facts_value
    

st.session_state.form_mode = None
if 'seed_prefl' not in st.session_state:
   st.session_state.seed_prefl = 0
    
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
        #seed_prefl = 0
        st.session_state.seed_prefl = random.randint(0, 4000)
        st.write(st.session_state.seed_prefl)        
    
    if st.session_state.form_mode == "Prefilled Form":
        st.write(st.session_state.seed_prefl)
        first_party_value, second_party_value, issue_area_value, case_facts_value = prefilled_form(st.session_state.seed_prefl)
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
            prediction = predict_party(case_facts_input)
            if prediction == 'First Party':
                #st.write(f'The judgment is likely to favor {first_party_input}.')
                st.markdown(f"<h1 style='text-align:center; color: green;'> The judgment is likely to favor {first_party_input}. </h1>", unsafe_allow_html=True)
            else:
                #st.write(f'The judgment is likely to favor {second_party_input}.')
                st.markdown(f"<h1 style='text-align:center; color: green;'> The judgment is likely to favor {second_party_input}. </h1>", unsafe_allow_html=True)
 
       
if __name__ == '__main__':
    main()


#This code loads Keras LSTM model and tokenizer, defines a function to make predictions based on case facts, and defines your Streamlit app with two sections: "Case Information" and "Prediction". You can customize this code to fit your specific use case.