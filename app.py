
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the summarizer model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Function to summarize
def summarize_email(email_body):
    input_text = "summarize: " + email_body
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Streamlit interface
st.title("📧 Email Summarizer App")
email_input = st.text_area("Paste your email content here:")

if st.button("Summarize"):
    if email_input.strip() != "":
        summary = summarize_email(email_input)
        st.success("📝 Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some email content first.")
