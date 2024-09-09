# -*- coding: utf-8 -*-
"""imagecrop.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Z72MParhxEH1ewAkWQNx12MrL3lYGiF4
"""
# Import necessary libraries
import openai
import time
import os
import io
from openai import OpenAI
import streamlit as st
from PIL import Image
import base64
import numpy as np
import pyttsx3
import tempfile
import fitz
from streamlit_cropper import st_cropper
from sec_key1 import my_sk
from io import BytesIO
import asyncio
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
# from streamlit_drawable_canvas import st_canvas
import pytesseract
import cv2
from tensorflow.keras.models import load_model
from pdf2image import convert_from_path  # Library to convert PDF to images
from pdf2image import convert_from_path
from IPython.display import display


# Initialize session state
# Import necessary libraries
import openai
import os
import io
from openai import OpenAI
import streamlit as st
from PIL import Image, ImageEnhance
import base64
import numpy as np
import pyttsx3
import tempfile
import fitz
from streamlit_cropper import st_cropper
from sec_key1 import my_sk
from io import BytesIO
import asyncio
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
# from streamlit_drawable_canvas import st_canvas
import pytesseract
import cv2
from tensorflow.keras.models import load_model
from pdf2image import convert_from_path  # Library to convert PDF to images
from pdf2image import convert_from_path
from IPython.display import display




#function to search for bill pdfs from desired location

def search_pdf_path(customer_id):
    search_directory = "C:\\Users\\mz7505\\Downloads"  # Update this to the directory where PDFs are stored
    for root, dirs, files in os.walk(search_directory):
        for file in files:
            if file.startswith(str(customer_id)) and file.endswith(".pdf"):
                return os.path.join(root, file)
    return None

#function to convert the bill pdf to JPEG

def convert_pdf_to_images(pdf_file_path):
    doc = fitz.open(pdf_file_path)
    image_buffers = []

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        
        # Convert to image
        image_buffer = pix.tobytes("jpeg")
        image_buffers.append(image_buffer)
        
        # Save image
        with open(f"page_{page_num + 1}.jpeg", "wb") as img_file:
            img_file.write(image_buffer)
    
    return image_buffers

#Function to convert the uploaded image into str"
def encode_image(file):
    image_data = file.read()
    return base64.b64encode(image_data).decode("utf-8")



# OpenAI LLM call to work with images, prompt, and contextual content

def ask_get_answer(prompt, image_file, contextual_data):
    Model = "gpt-4o"
    openai.api_key = my_sk
    base64_image = encode_image(image_file)
    response = openai.chat.completions.create(
        model=Model,
        messages=[
            {"role": "system", "content": "Your expertise is in demand for image-driven inquiries. Utilize the provided image, user prompt and context documents to address user questions exclusively. Keep responses focused only on the user query, refraining from additional information. Strictly refrain using words from where this content is being retrieved, such as 'According to the provided context', 'provided content' & 'Additional content'." },
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }},
                {"type": "text", "text": f"Additional Content: {contextual_data}"}
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

def display_selected_image(selected_image):
    
       
    # Initialize session state for cropped image and prompt
    if 'cropped_img' not in st.session_state:
        
        st.session_state.cropped_img = None
        
    if 'prompt' not in st.session_state:
        
        st.session_state.prompt = ""
        
    if 'cropped_img_buffer' not in st.session_state:
        
        st.session_state.cropped_img_buffer = BytesIO()
        
    if 'option' not in st.session_state:
        
        st.session_state.option = "No"  # Default value   
    
    
    img1 = Image.open(BytesIO(selected_image))
    
    #st.image(img1, use_column_width=True) #caption="Selected Page for Clarification", 
    
    # Integrate cropping tool here
    cropped_img = st_cropper(img1, realtime_update=True, box_color='#0000FF', aspect_ratio=None)
    
    # Process the cropped image
    if st.button("Select to pinpoint your area of interest!", key="crop_button"):
        
        st.write("Fantastic!")
        
        _ = cropped_img.thumbnail((150,150))
        
        enhancer = ImageEnhance.Sharpness(cropped_img)
        
        cropped_img = enhancer.enhance(6.0)
        
        st.image(cropped_img, caption="I'm here to help you unlock the insights hidden within your image.")
        
        # Save the cropped image to session state
        st.session_state.cropped_img = cropped_img
        
        # Save the cropped image to a BytesIO object
        st.session_state.cropped_img_buffer = BytesIO()
        
        st.session_state.cropped_img.save(st.session_state.cropped_img_buffer, format='PNG')
        
        st.session_state.cropped_img_buffer.seek(0)
        
        
        # Save the original image to a BytesIO object
        img_buffer = BytesIO()
        
        img1.save(img_buffer, format='PNG')
        
        img_buffer.seek(0)
        
        # Set this flag to True when the ROI is selected
        st.session_state.roi_selected = True
        
    if st.session_state.get('roi_selected'):
        
        # Handling the option to describe the selected ROI
        option = st.radio("You can select 'Yes' if you would like to describe more about the selected Region of Interest", ["Yes", "No"],
                         index=0 if st.session_state.option == "Yes" else 1,key="roi_radio")
        
#         st.session_state.option = option  # Update session state
        
        
          
        
        
        # Handling the option to describe the selected ROI
#         if 'option' not in st.session_state:
            
#             st.session_state.option = "No"  # Default value
        
#         option = st.radio("You can select 'Yes' if you would like to describe more about the selected Region of Interest", ["Yes", "No"],
#                          index=0 if st.session_state.option == "Yes" else 1,key="roi_radio")
        
        st.session_state.option = option  # Update session state
        
        if st.session_state.option == 'Yes':
            
            # Initialize the text input area only if 'Yes' is selected
            if 'prompt_input' not in st.session_state:
                
                st.session_state.prompt_input = ""
            
            prompt = st.text_area("Would you like to provide more details about the image to expedite our progress?:",value=st.session_state.prompt_input,key="prompt_input_text")
            
            st.session_state.prompt_input = prompt
#             if prompt:
            
            if st.session_state.prompt_input:
            
#             if st.session_state.prompt:
                
                
                content = '''This account was opened on 01/01/2023. The main account number is 177100146242 and the foundation account is 03008387. 
                $162.03 is the total billed amount till the month of April 14th, 2023. This $162.03 is charges against each number (682.715.3034 is $13.54,817.228.8578 is $ 35.87,817.371.8751 is $35.87,817.615.0605 is $40.88,817.944.2571 is $35.87).
                'Michael Headrick' has only one number (682.715.3034) and is the primary account owner.
                The total plan cost for five numbers (682.715.3034,817.228.8578,817.371.8751,817.615.0605,817.944.2571) is $130, where as the average across five numbers is $26. 
                The Total cost across five numbers (682.715.3034,817.228.8578,817.371.8751,817.615.0605,817.944.2571) is $162.03 and the average cost across five numbers is $32.40. 
                'Nancy Headrick' is the secondary owner who has four lines (817.228.8578,817.371.8751,817.615.0605,817.944.2571). The total plan cost for 'Nancy Headrick' who has four lines is $120 where each line number has a charge of $30 and average plan cost for four lines is $30. Whereas for 'Nancy Headrick' four lines the total billed cost is $148.49 and its corresponding average is $37.12. This is the usage cost of each number assigned to "Nancy Headrick" (817.228.8578 is $35.87,817.371.8751 is $35.87,817.615.0605 is $40.88,817.944.2571 is $35.87). 
                The add-ons $4.99 were added to 'Nancy Headrick' number '817.615.0605'. The add-ons is related to 'AT&T Secure Family', a wireless add-on added on 01/04/2023. 
                $22.88 is releated to AT&T fee and surcharges, This $22.88 is the sum of five numbers, this is the split of the surcharges (682.715.3034) which belongs to 'Michael Headrick' is $3.54, 'Nancy Headrick' four lines are charged $4.83 each and totals upto $19.32. The average surcharge for 'Nancy Headrick' numbers is $4.83, 
                whereas $4.16 is related to Government fees and taxes, which is only applicable to 'Nancy Headrick' four numbers, each number being charged $1.04.'''
                

                if st.button("Run!"):
                    
#                     st.write("Thanks for your query")
                    
                    with st.spinner("Unveiling the details..."):
                        
                        answer = ask_get_answer(st.session_state.prompt_input, st.session_state.cropped_img_buffer, content)
                        
                        st.text_area("Voila!Here you go: ", value=answer)
            
        else:
            
            st.warning("Oops! No prompt selected")





# Function to display the bill PDF pages as images with selection option
def display_images_with_selection(image_buffers):
    
    st.write("Here is your Bill Pages:")
    
    # Initialize session state variables
    if 'selected_image' not in st.session_state:
    
        st.session_state.selected_image = None
        
        st.session_state.selected_page_number = None
    
    

    for i, buffer in enumerate(image_buffers):
        
        # Convert the image buffer back to an image
        img = Image.open(BytesIO(buffer))
        
        # Display the image in Streamlit with a checkbox to select
        if st.checkbox(f"Select Page {i+1}", key=f"select_{i}"):
            
            st.session_state.selected_image = buffer  # Save the selected image buffer
            
            st.session_state.selected_page_number = i + 1  # Save the selected page number

        st.image(img, caption=f"Page {i+1}", use_column_width=True)

    # Display the selected image after selection
    if st.session_state.selected_image:
        
        st.write(f"You have selected Page {st.session_state.selected_page_number} for clarification.")
        
        display_selected_image(st.session_state.selected_image)
        

# Function to display the selected image after selection



            
def model_response(user_prompt):
    Model = "gpt-4o"
    openai.api_key = my_sk
    
    response = openai.chat.completions.create(
    model=Model,
    messages=[
                {"role": "system", "content": "You are a helpful billing assistant that responds to the user query. You are responsible to empathize whenever required and get the cust_id to assist better."},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt}
                ]}
            ],
    temperature=0.0
        )
    
    assistant_response = response.choices[0].message.content
    return assistant_response
  
def welcomebot():
    
    # Initialize session state for user_prompt and customer_id
    if 'user_prompt' not in st.session_state:
        
        st.session_state.user_prompt = ""
        
    if 'assistant_response' not in st.session_state:
        
        st.session_state.assistant_response = ""

    if 'customer_id' not in st.session_state:
        
        st.session_state.customer_id = ""
    
    logo_img = Image.open('Color-ATT-Logo.png')
    
    st.image(logo_img, width=250,use_column_width=False)
    
    st.title('Talking with your AT&T Billing Genie!!..')
    
    st.subheader('Lets spin the Magic!...')
    
    with st.sidebar:
        
        st.image('ATT_side.png', width=300, use_column_width=False) 
        
    
#     st.title("Billing Genie - Welcome Chatbot")
    st.write("Hi, this is Billing Genie! How can I assist you today?")
    
    if "message" not in st.session_state:
        st.session_state.message=[]
        
    for message in st.session_state.message:
        with st.chat_message(message["role"]):
            st.markdown(message['content'])
    
    
    
    
    if user_prompt:= st.chat_input("Ask your question"):
        
        # Save user prompt to session state
        st.chat_message("User").markdown(user_prompt)
        st.session_state.message.append({"role":"User","content": user_prompt})
        if user_prompt.isnumeric():
            try:
                # Save customer ID to session state
                st.session_state.customer_id= int(user_prompt)
                
                st.write(f"Thank you!. Fetching your bill details...")
                
                pdf_path = search_pdf_path(st.session_state.customer_id)
                
                if pdf_path:
                    
#                     st.write(f"Found PDF for Customer ID {customer_id}: {pdf_path}")

                    image_buffers = convert_pdf_to_images(pdf_path)
    
                    display_images_with_selection(image_buffers)
                    
                
                else:
                
                    st.write(f"No PDF found for Customer ID {st.session_state.customer_id}.")
                    
            except ValueError:
                
                st.write("Invalid input. Please enter a valid numeric Customer ID.")
        
    #     base64_image = encode_image(image_file)
        else:
        
        
            # Save assistant response to session state
            st.session_state.assistant_response = model_response(user_prompt)
        
            st.write("Billing Genie:", st.session_state.assistant_response)
        
        #customer_id = st.text_input("Can I have your Cust_ID to assist you better? Please enter a number:")
        
        
         
            
  
    # Run the Streamlit app
if __name__ == "__main__":
    #asyncio.run(welcomebot())
    welcomebot() 