import streamlit as st
import mysql.connector
from mysql.connector import Error

def create_connection():
    """ Create a database connection to the MySQL database """
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',      # e.g., 'localhost'
            user='root',           # e.g., 'root'
            password='',           # your database password
            database='phuean'
        )
        if connection.is_connected():
            st.success("Connection to MySQL DB successful")
    except Error as e:
        st.error(f"The error '{e}' occurred")
    
    return connection

def login():
    st.title("Welcome to :red[Phuean Jai] Speech to Tone")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            query = "SELECT * FROM users WHERE username = %s AND password = %s"
            cursor.execute(query, (username, password))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            if result:
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.success("Login successful")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
