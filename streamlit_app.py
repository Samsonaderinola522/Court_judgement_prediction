import streamlit as st
import random

def main():
    st.title('Random Number Generator')
    cvu = "x"
    # Add a text input
    number_input = st.text_input('Enter a number:')
    
    # Add a text input
    number_input_new = st.text_input('updated', value=cvu)
    
    # Add a button that generates a random number
    if st.button('Generate Random Number'):
        # Generate a random integer between 1 and 10
        random_number = random.randint(1, 10)
        
        # Update the value of the text input with the generated number
        st.session_state.cvu = str(random_number)

if __name__ == '__main__':
    main()