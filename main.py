import streamlit as st
from streamlit_option_menu import option_menu
from login import login
from register import register
from deploy import deploy  
from profiles import profiles  

def main():
    st.set_page_config(page_title="Phuean Jai", page_icon=":smiley:", layout="wide")

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    with st.sidebar:
        st.image("image/speechlogo.png", use_column_width=True)
        app = option_menu(
            menu_title='Phuean Jai',
            options=['Home', 'Account', 'Profile'],  
            icons=['house', 'person', 'list-task'],
            default_index=0,
        )

    if app == "Home":
        st.markdown(
            """
            <style>
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .centered {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 80vh;
                flex-direction: column;
                animation: fadeIn 2s ease-in-out;
            }
            .title {
                color: #ff9d47;
                font-weight: bold;
                font-size: 2.5rem;
            }
            .subtitle {
                font-size: 1.25rem;
                color: #444;
            }
            .center-img {
                display: flex;
                justify-content: center;
                align-items: center;
                width: 100%;
                margin-bottom: 20px;
            }
            img.center {
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 50%;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .icon {
                margin-top: 20px;
                font-size: 3rem;
                color: #ff6347;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        # Centered content
        st.markdown(
            """
            <div class="centered">
                <div class="center-img">
                    <img src="http://localhost:8501/media/444e70c5cb568bd1ed4a1c5c6519337cf77963f4ede08772405c2377.png" style="max-width: 800px; height: auto;">
                </div>
                <h2 class="title">Welcome to the Phuean Jai</h2>
                <p class="titles">The development of the "Phuean Jai Speech to tone" model focuses on classifying
                speech attitudes using tone and frequency of speech. The main principle of operation of the
                system is to detect and analyze the tone of the voice that appears during a person's speech.
                Including examining changes in voice frequency and tone of speech in different contexts. The
                models used are EfficientNet b7, CNN and LSTM which are designed to accurately identify the
                sentiment and attitude of speech. EfficientNet b7's accuracy was 73%, CNNs 94% accurate, and
                LSTMs 42%, indicating the system's potential and efficiency in classifying attitudes from speech
                with a wide range of tones and frequencies. They speak differently in real situations.</p>
                <i class="fas fa-volume-up icon"></i>
            </div>
            """,
            unsafe_allow_html=True
        )

    elif app == "Account":
        if st.session_state['logged_in']:
            # Assuming you have a way to retrieve the username of the logged-in user
            deploy(st.session_state['username'])
        else:
            with st.sidebar.expander("Login/Register"):
                choice = st.radio("Select Option", ["Login", "Register"])

            if choice == "Login":
                login()
                if st.session_state['logged_in']:
                    st.experimental_rerun()  # Refresh the app to reflect the login status
            elif choice == "Register":
                register()

    elif app == "Profile":
        if st.session_state['logged_in']:
            profiles(st.session_state['username'])
            if st.button("Logout"):
                st.session_state['logged_in'] = False
                st.experimental_rerun()  # Refresh the app to reflect the logout status
        else:
            st.markdown(
                "<h1 style='text-align: center;'>Please login to view your profile.</h1>",
                unsafe_allow_html=True
            ) 
if __name__ == "__main__":
    main()