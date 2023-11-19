from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text