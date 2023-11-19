
import os
import re
import nltk
import torch
import gensim
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel
from summa.summarizer import summarize
from summa import keywords
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from nltk.tokenize import sent_tokenize
from torch.nn.functional import cosine_similarity
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def genarate_bert(model_passed, tokenizer_passed, input_passed, input_length):
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_passed)
   
    model =  BertForSequenceClassification.from_pretrained(model_passed)
    model = model.bert

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    def bert_sentence_embeddings(text, tokenizer, model, device, window_size=128, stride=64):
        tokens = tokenizer(text, return_tensors='pt', add_special_tokens=True, padding=False, truncation=False)
        all_embeddings = []
        for i in range(0, tokens.input_ids.size(1), stride):
            window_range = slice(i, min(i + window_size, tokens.input_ids.size(1)))
            window_tokens = {key: value[:, window_range].to(device) for key, value in tokens.items()}
            with torch.no_grad():
                outputs = model(**window_tokens)
            all_embeddings.append(outputs.last_hidden_state.mean(dim=1))
        return torch.mean(torch.cat(all_embeddings, dim=0), dim=0).squeeze()

    def extractive_summarization(document_text, model, tokenizer, summary_length, device, batch_size=4):
        sentences = sent_tokenize(document_text)
        sentence_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_embeddings = [bert_sentence_embeddings(sentence, tokenizer, model, device) for sentence in batch_sentences]
            sentence_embeddings.extend(batch_embeddings)
        doc_embedding = torch.mean(torch.stack(sentence_embeddings), dim=0)
        scores = []
        for emb in sentence_embeddings:
            score = cosine_similarity(emb.unsqueeze(0), doc_embedding.unsqueeze(0)).item()
            scores.append(score)
        top_sentence_indices = np.argsort(scores)[-summary_length:]
        summary = ' '.join([sentences[i] for i in sorted(top_sentence_indices)])
        return summary
    
    document_text = input_passed
    new_length = input_length // 6
    summary_passed = extractive_summarization(document_text, model, tokenizer, new_length, device = device)
    return summary_passed; 