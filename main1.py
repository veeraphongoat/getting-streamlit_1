import streamlit as st
import mysql.connector
from mysql.connector import Error

# ฟังก์ชั่นเพื่ออ่านไฟล์รูปภาพ
def convert_to_binary_data(file):
    binary_data = file.read()
    return binary_data

# การเชื่อมต่อกับฐานข้อมูล MySQL
def create_connection():
    """ Create a database connection to the MySQL database """
    connection = None
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='phuean'
        )
        if connection.is_connected():
            st.success("Connection to MySQL DB successful")
    except Error as e:
        st.error(f"The error '{e}' occurred")
    
    return connection

# ฟังก์ชั่นสำหรับการสร้างตาราง
def create_images_table(connection):
    cursor = connection.cursor()
    query = """
    CREATE TABLE IF NOT EXISTS images (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            image LONGBLOB NOT NULL
    )
    """
    cursor.execute(query)
    connection.commit()
    cursor.close()

# การอัพโหลดรูปภาพ
def insert_image(name, binary_image):
    connection = create_connection()
    cursor = connection.cursor()
    query = "INSERT INTO images (name, image) VALUES (%s, %s)"
    cursor.execute(query, (name, binary_image))
    connection.commit()
    cursor.close()
    connection.close()

# ส่วนของ Streamlit สำหรับอัพโหลดไฟล์
st.title("Upload Image to MySQL")

# สร้างตาราง images ถ้ายังไม่มี
connection = create_connection()
create_images_table(connection)
connection.close()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_name = uploaded_file.name  # ดึงชื่อไฟล์จาก uploaded_file
    binary_image = convert_to_binary_data(uploaded_file)
    insert_image(file_name, binary_image)  # ใช้ชื่อไฟล์เป็นพารามิเตอร์
    st.success("Image uploaded successfully!")
