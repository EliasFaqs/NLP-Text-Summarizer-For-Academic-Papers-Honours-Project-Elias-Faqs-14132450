import tkinter as tk
from tkinter import filedialog, Text, messagebox, StringVar, OptionMenu
from tkinter.ttk import Progressbar
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForSequenceClassification
from PyPDF2 import PdfReader, PdfWriter
import os
from model_utils import select_model_and_dataset
from Pegasus_generate_summary import genarate_pegasus
from BERT_generate_summary import genarate_bert
from transformers import PegasusForConditionalGeneration

# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text



def get_summary_from_model(input_text, summary_length, summary_method, dataset):
    model, tokenizer = select_model_and_dataset(summary_method, dataset)
    summary = ""  # 

    if summary_method == "pegasus":
        summary = genarate_pegasus(model, tokenizer, input_text, summary_length)
    elif summary_method == "bert":  
        summary = genarate_bert(model, tokenizer, input_text, summary_length)
   
    return summary

def save_summary_to_pdf(text, original_path):
    output_filename = os.path.splitext(original_path)[0] + "_Summary.pdf"
    with open(output_filename, "wb") as output_file:
        pdf_writer = PdfWriter()
        pdf_writer.add_page(text)
        pdf_writer.write(output_file)
    return output_filename

def load_pdf():
    global pdf_path
    pdf_path = filedialog.askopenfilename()
    if pdf_path:
        text = extract_text_from_pdf(pdf_path)
        input_field.delete("1.0", tk.END)
        input_field.insert(tk.END, text)

def summarize_text():
    progress_bar.grid(row=6, column=0, pady=10, padx=5, columnspan=2, sticky=tk.W+tk.E)
    root.update()

    input_text = input_field.get("1.0", tk.END)
    if not input_text:
        return

    summary = get_summary_from_model(input_text, summary_length_slider.get(), summary_method.get(), dataset_var.get())

    output_field.delete(1.0, tk.END)
    output_field.insert(tk.END, summary)

    # Check if save_to_pdf is checked
    if save_to_pdf_var.get():
        save_path = save_summary_to_pdf(summary, pdf_path)
        messagebox.showinfo("Info", f"Summary saved to: {save_path}")

    progress_bar.grid_remove()

root = tk.Tk()
root.title("Document Summarizer")
root.geometry("1000x1000")

# Label with large, bold font at the top
title_label = tk.Label(root, text="Document Text Summarizer", fg="#1d3557", font=("Helvetica", 24, "bold"))
title_label.pack(side=tk.TOP, pady=(10, 20))  # Pack it at the top with some vertical padding for spacing

canvas = tk.Canvas(root, height=700, width=800, bg="#1d3557")
canvas.pack(fill=tk.BOTH, expand=True)

frame = tk.Frame(root, bg="#DDE2E5")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

load_button = tk.Button(frame, text="Load PDF", padx=10, pady=5, fg="white", bg="#1d3557", command=load_pdf)
load_button.grid(row=0, column=0, pady=10, padx=5, columnspan=2, sticky=tk.W+tk.E)

frame.columnconfigure(0, weight=2)

input_field = Text(frame, height=20, wrap=tk.WORD)
input_field.grid(row=1, column=0, pady=10, padx=5, columnspan=2, sticky=tk.W+tk.E)

summary_method = tk.StringVar(value="Abstractive")
r1 = tk.Radiobutton(frame, text="Abstractive (Pegasus)", variable=summary_method, value="pegasus", bg="#DDE2E5")
r2 = tk.Radiobutton(frame, text="Extractive (BERT)", variable=summary_method, value="bert", bg="#DDE2E5")

r1.grid(row=2, column=0, pady=10)
r2.grid(row=2, column=1, pady=10)

frame.columnconfigure(0, weight=1)
frame.columnconfigure(1, weight=1)

dataset_var = StringVar(root)
dataset_var.set("arxiv")  # default value
dataset_label = tk.Label(frame, text="Dataset:", bg="#DDE2E5")
dataset_label.grid(row=3, column=0, pady=10, sticky=tk.W)

dataset_menu = OptionMenu(frame, dataset_var, "arxiv", "pubmed")
dataset_menu.grid(row=3, column=1, pady=10, sticky=tk.W+tk.E)



summary_length_label = tk.Label(frame, text="Summary Length:", bg="#DDE2E5")
summary_length_label.grid(row=4, column=0, pady=10, sticky=tk.W)

summary_length_slider = tk.Scale(frame, from_=30, to=512, orient=tk.HORIZONTAL, bg="#DDE2E5")
summary_length_slider.set(256)
summary_length_slider.grid(row=4, column=1, pady=10, sticky=tk.W+tk.E)

summarize_button = tk.Button(frame, text="Summarize", padx=10, pady=5, fg="white", bg="#1d3557", command=summarize_text)
summarize_button.grid(row=5, column=0, pady=10, padx=5, columnspan=2, sticky=tk.W+tk.E)

# Checkbox for saving summary to PDF
save_to_pdf_var = tk.IntVar()
save_to_pdf_checkbox = tk.Checkbutton(frame, text="Save summary to PDF", variable=save_to_pdf_var, bg="#DDE2E5")
save_to_pdf_checkbox.grid(row=6, column=0, pady=10, padx=5, columnspan=2, sticky=tk.W+tk.E)

# Progress bar
progress_bar = Progressbar(frame, orient=tk.HORIZONTAL, length=100, mode='indeterminate')

output_field = Text(frame, height=10, wrap=tk.WORD)
output_field.grid(row=8, column=0, pady=10, padx=5, columnspan=2, sticky=tk.W+tk.E)

root.mainloop()