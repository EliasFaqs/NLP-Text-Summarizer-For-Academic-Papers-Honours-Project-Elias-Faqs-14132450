from transformers import PegasusForConditionalGeneration

def select_model_and_dataset(model_name, dataset_name):
    if model_name == "bert" and "dataset_name " == "arxiv":
        # ARXIV and BERT
        model = ".\BERT_Files\BERT_arxiv\finetuned_arxiv_bert-base-uncased"
        tokenizer = ".\BERT_Files\BERT_arxiv\finetuned_arxiv_bert-base-uncased"
        return model, tokenizer

    elif model_name == "pegasus" and dataset_name == "arxiv":
        # ARXIV and Pegasus
        model = ".\Pegasus_Files\pegasus_arxiv\tokenized_google_pegasus-arxiv"
        tokenizer = ".\Pegasus_Files\pegasus_arxiv\tokenized_google_pegasus-arxiv"
        return model, tokenizer

    elif model_name == "bert" and dataset_name == "pubmed":
       #Pubmed and BERT
        model = ".\BERT_Files\BERT_pubmed\finetuned_pubmed_bert-base-uncased"
        tokenizer = ".\BERT_Files\BERT_pubmed\finetuned_pubmed_bert-base-uncased"
        return model, tokenizer

    elif model_name == "pegasus" and dataset_name == "pubmed" :
       #Pubmed and Pegasus
        model = ".\Pegasus_Files\pegasus_pubmed\finetuned_pubmed_google_pegasus-pubmed"
        tokenizer = ".\Pegasus_Files\pegasus_pubmed\finetuned_pubmed_google_pegasus-pubmed"
        return model, tokenizer

    else:
        
        model = "Unknown_Model"
        tokenizer = "Unknown_Tokenizer"
        return model, tokenizer