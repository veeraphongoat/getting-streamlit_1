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

def create_users_table(connection):
    cursor = connection.cursor()
    query = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        email VARCHAR(255) NOT NULL,
        password VARCHAR(255) NOT NULL
    )
    """
    cursor.execute(query)
    connection.commit()
    cursor.close()

def insert_user(connection, user):
    cursor = connection.cursor()
    query = """
    INSERT INTO users (username, email, password) 
    VALUES (%s, %s, %s)
    """
    cursor.execute(query, user)
    connection.commit()
    cursor.close()

def register():
    st.title("Welcome :red[Phuean Jai] Speech to tone")
    connection = create_connection()
    if connection:
        create_users_table(connection)

    with st.form("register_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        submit_button = st.form_submit_button(label="Register")

    if submit_button:
        if len(password) < 6:
            st.error("Password must be at least 6 characters long")
        else:
            if connection:
                user = (username, email, password)  # In a real application, ensure passwords are hashed
                try:
                    insert_user(connection, user)
                    st.success("Registration successful!")
                    st.info("Please login with your new credentials.")
                except Error as e:
                    st.error(f"The error '{e}' occurred while inserting data")
                finally:
                    connection.close()
            else:
                st.error("Failed to connect to the database")
