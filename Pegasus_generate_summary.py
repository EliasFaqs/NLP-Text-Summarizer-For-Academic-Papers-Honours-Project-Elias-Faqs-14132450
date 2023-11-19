
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments

def genarate_pegasus(model_passed, tokenizer_passed, input_passed, length_passed):
    
    model_path = model_passed
    tokenizer_path = tokenizer_passed
    model = PegasusForConditionalGeneration.from_pretrained(model_path)
    tokenizer = PegasusTokenizer.from_pretrained(tokenizer_path)

    from transformers import PegasusTokenizer, PegasusForConditionalGeneration

    def postprocess_summary(summary):
        redundant_phrases = ["In this paper,", "In this work"]
        for phrase in redundant_phrases:
            summary = summary.replace(phrase, "")
        return summary.strip()

    def summarize_segment(segment, model, tokenizer, max_length=1024, min_length=64):
        tokens = tokenizer(segment, truncation=True, padding="longest", return_tensors="pt")
        summary_ids = model.generate(tokens["input_ids"], num_beams=4, max_length=max_length, min_length=min_length, length_penalty=2.0, no_repeat_ngram_size=3, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return postprocess_summary(summary)

    def count_words(text):
        return len(text.split())

    def sliding_window_summarization(document_text, summary_length=256, window_size=600, step_size=300):
        # Use the model and tokenizer passed as arguments
        tokens = tokenizer(document_text, return_tensors="pt", truncation=False)["input_ids"][0]

        summarized_text = ""
        words_count = 0
        for i in range(0, len(tokens), step_size):
            if words_count >= summary_length:
                break
            window_end = min(i + window_size, len(tokens))
            window = tokens[i:window_end]
            segment_text = tokenizer.decode(window, skip_special_tokens=True)
            segment_summary = summarize_segment(segment_text, model, tokenizer)
            segment_word_count = count_words(segment_summary)
            if words_count + segment_word_count > summary_length:
                break
            summarized_text += segment_summary + " "
            words_count += segment_word_count

        return summarized_text.strip()

    summary = sliding_window_summarization(input_passed, length_passed)  
    return summary