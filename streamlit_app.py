# import streamlit as st
# import fitz  # PyMuPDF

# # Function to extract text from a specific page of a PDF
# def extract_text_from_pdf(pdf_path, page_number):
#     # Open the PDF file
#     document = fitz.open(pdf_path)
    
#     # Get the specific page
#     page = document.load_page(page_number)
    
#     # Extract text from the page
#     text = page.get_text("text")
    
#     return text

# def extract_all_text_from_pdf(pdf_path, output_txt_path):
#     # Open the PDF file
#     document = fitz.open(pdf_path)
    
#     # Initialize a string to hold all text
#     all_text = ""
    
#     # Iterate through all the pages
#     for page_number in range(len(document)):
#         # Get the specific page
#         page = document.load_page(page_number)
        
#         # Extract text from the page
#         text = page.get_text("text")
        
#         # Append the text to the all_text string
#         all_text += text + "\n"  # Add a newline after each page's text
    
#     return all_text

# def main():
#     st.title("PDF Viewer with LLM")

#     # Layout: PDF Viewer and Input Fields Side by Side
#     col1, col2 = st.columns(2)

#     with col1:
#         pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
#         print(pdf_file)

#         st.subheader("Choose what you want us to do...........")
                
#         task_option = st.selectbox(
#             "Choose a task",
#             ("Summarization", "Search Similarity", "Question and Answer")
#         )

#         if pdf_file is not None:
#             pdf_path = pdf_file.name
#             with open(pdf_path, "wb") as f:
#                 f.write(pdf_file.getbuffer())


#             page_number = st.number_input("Page number", min_value=0, max_value=100, value=0, step=1)
#             text = extract_text_from_pdf(pdf_path, page_number)
#             print('Text ',text)

#             doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
#             with col2:
#                 page_num = st.number_input("Page Number", min_value=1, max_value=len(doc), value=1)
#                 page = doc.load_page(page_num - 1)
#                 pix = page.get_pixmap()
#                 img = pix.tobytes("png")
                
#                 st.image(img, caption=f'Page {page_num}', use_column_width=True)


#     question = st.text_input("Question")
#     prompt_style = st.text_input("Prompt Style")
#     #prompt_style = st.text_area("Prompt Styles")
#     generate_button = st.button("Generate Response")

#     if generate_button and question and prompt_style:
#         pass
#             # if task_option == "Summarization":
#             # response = openai.Completion.create(
#             #     engine="text-davinci-003",  # Replace with appropriate model
#             #     prompt=f"Summarize the following text:\n\n{question}\n\n{prompt_style}",
#             #     max_tokens=150
#             # )
#             # elif task_option == "Search Similarity":
#             #     response = openai.Completion.create(
#             #         engine="text-davinci-003",  # Replace with appropriate model
#             #         prompt=f"Find similar text to the following:\n\n{question}\n\n{prompt_style}",
#             #         max_tokens=150
#             #     )
#             # elif task_option == "Question and Answer":
#             #     response = openai.Completion.create(
#             #         engine="text-davinci-003",  # Replace with appropriate model
#             #         prompt=f"{prompt_style}\n\n{question}",
#             #         max_tokens=150
#             #     )
        

import streamlit as st
import fitz  # PyMuPDF


st.markdown("""
    <style>
    .container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .content {
        width: 80%;
    }
    </style>
    <div class="container">
        <div class="content">
        </div>
    </div>
    """, unsafe_allow_html=True)

# Function to extract all text from a PDF
def extract_all_text(pdf_path):
    document = fitz.open(pdf_path)
    all_text = ""
    for page_number in range(len(document)):
        page = document.load_page(page_number)
        text = page.get_text("text")
        all_text += text + "\n"
    return all_text

# Function to extract text from specific pages of a PDF
def extract_specific_pages_text(pdf_path, page_numbers):
    document = fitz.open(pdf_path)
    pages_text = []
    for page_number in page_numbers:
        page = document.load_page(page_number)
        text = page.get_text("text")
        pages_text.append((page_number + 1, text))
    return pages_text

def render_page_as_image(pdf_document, page_number):
    from PIL import Image
    page = pdf_document.load_page(page_number)
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def main():
    st.title("RAG Text Summarization and Question Answering")
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    col1, col2 = st.columns((3,3))
    all_text=""
    text_per_page=""
    with col1:
        if pdf_file:
            # Save uploaded PDF to a temporary file
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.getbuffer())
            
            document = fitz.open("temp.pdf")
            total_pages = len(document)

            # Option to choose extraction method
            option = st.selectbox(
                "Choose page to work on",
                ("All pages", "Specific pages")
            )
            
            if option == "All pages":
                #st.write("Extracting all text from the PDF...")
                all_text = extract_all_text("temp.pdf")

            elif option == "Specific pages":
                page_numbers = st.multiselect(
                    "Select pages to extract",
                    options=list(range(total_pages)),
                    format_func=lambda x: f"Page {x + 1}"
                )
                
                if page_numbers:
                    #st.write(f"Extracting text from pages: {', '.join([str(p + 1) for p in page_numbers])}")
                    text_per_page = extract_specific_pages_text("temp.pdf", page_numbers)
                else:
                    st.write("Please select at least one page.")

    if pdf_file:
        response=""
        with col2:
            st.title("PDF VIEWER")
            # Render selected page as image
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            page_num = st.number_input("Page Number", min_value=1, max_value=len(doc), value=1)
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap()
            img = pix.tobytes("png")
            st.image(img, caption=f'Page {page_num}', use_column_width=True)

        with col1:
            task_option = st.selectbox(
                    "Choose a task",
                    ("Summarization", "Question and Answer")
                )
            prompt_style = st.text_area("Prompt Styles")
            if task_option=="Question and Answer":
                question = st.text_area("Question")

            generate_button = st.button("Generate Response")
            text=""
            if generate_button and prompt_style:               
                if len(all_text.split(' '))>10:
                    text=all_text

                if isinstance(text_per_page,list):
                    text=text_per_page[0][1]  #list, tuple , second 
                
                if task_option == "Summarization":
                    with st.spinner('Summarization...'):
                        from All_task import Tasks
                        summ_obj=Tasks(text)
                        response=summ_obj.summarize_text(prompt_style)
                elif task_option == "Question and Answer":
                    if question:
                        with st.spinner('Answering...'):
                            from All_task import Tasks
                            summ_obj=Tasks(text)
                            response=summ_obj.qa_task(question,prompt_style)
        with st.container():
            st.info(response)

if __name__=="__main__":
    main()

