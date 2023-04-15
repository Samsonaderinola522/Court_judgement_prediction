import streamlit as st

def display_text():
    st.write('Fill this form')
    Gun = st.text_input('what is gun?')

def display_text2():
    st.write('Fill this form')
    Gun = st.text_input('what is gun?', value=Ak)

def main():
    st.title('My App')
    
    Ak = "Ak 47"
    # Display a button to refresh the text
    if st.button('Refresh text'):
        display_text()

    # Display a button to refresh the chart
    if st.button('Refresh text2'):
        display_text2()

if __name__ == '__main__':
    main()