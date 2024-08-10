import streamlit as st
import mysql.connector
from mysql.connector import Error
from io import BytesIO

def create_connection():
    """Create a database connection to the MySQL database"""
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',  # Replace with your MySQL password
            database='phuean'
        )
        if connection.is_connected():
            st.success("Connection to MySQL DB successful")
    except Error as e:
        st.error(f"The error '{e}' occurred")
    
    return connection

def fetch_user_audio_files(user_id):
    connection = create_connection()
    if connection:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT file_name, emotion_label, most_frequent_emotion FROM audio_files WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        result = cursor.fetchall()
        cursor.close()
        connection.close()
        return result
    return []

def profiles(username):
    st.title("User Profile")
    st.subheader(f"Username: {username}")

    connection = create_connection()
    if connection:
        cursor = connection.cursor()
        query = "SELECT id FROM users WHERE username = %s"
        cursor.execute(query, (username,))
        user_id = cursor.fetchone()
        cursor.close()
        connection.close()

        if user_id:
            user_id = user_id[0]  # Extracting user_id from tuple
            audio_files = fetch_user_audio_files(user_id)
            if audio_files:
                st.write("Uploaded/Recorded Audio Files and Predictions")
                
                # Creating a DataFrame from the fetched data
                import pandas as pd
                df = pd.DataFrame(audio_files)
                
                # Displaying the data in a table format
                st.table(df)
            else:
                st.write("No audio files uploaded or recorded yet.")
        else:
            st.error("User ID not found.")
    else:
        st.error("Error connecting to the database.")