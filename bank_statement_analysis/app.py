import streamlit as st
from typing import List
import pandas as pd
import os
from pydantic import BaseModel
from string import Template
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
import requests
import json

st.set_page_config(page_title="Bank Transaction Parser", layout="wide")

# Pydantic models
class AccountHolder(BaseModel):
    name: str
    account_number: str

class Transaction(BaseModel):
    date: str
    amount: float
    currency: str
    type: str
    description: str
    balance: float

class BankStatement(BaseModel):
    account_holder: AccountHolder
    transactions: List[Transaction]

# System templates
SYSTEM_MESSAGE = """
You are an advanced financial document parser specializing in extracting structured data from bank statements. Your task is to analyze the provided bank statement and convert the unstructured transaction data into a structured JSON format.

Key Requirements:
1. Extract account holder details
2. Parse all transactions accurately
3. Standardize date formats
4. Identify transaction types
5. Clean and simplify transaction descriptions
6. Handle multiple currencies
7. Maintain accurate balance tracking

Analyze the following bank statement and extract the required information:

Output Format:
Provide the extracted data in the following JSON structure:

{
    "transactions": {
        "account_holder": {
            "name": "Full Name",
            "account_number": "Complete Account Number"
        },
        "transactions": [
            {
                "date": "DD-MM-YYYY",
                "amount": float,
                "currency": "Currency Code",
                "type": "CREDIT or DEBIT",
                "description": "Cleaned Description",
                "balance": float
            },
            // ... more transactions
        ]
    }
}

Parsing Rules:
1. Account Holder:
   - Extract the full name as shown on the statement
   - Capture the complete account number

2. Transactions:
   - Date: Convert all dates to DD-MM-YYYY format
   - Amount: 
     - Remove currency symbols and commas
     - Convert to float
   - Currency: Detect the currency types INR or USD
   - Type:
     - CREDIT for deposits or positive amounts
     - DEBIT for withdrawals or negative amounts
   - Description:
     - Remove reference numbers, UPI IDs, and unnecessary banking terms
     - Keep relevant information like merchant names, payment purposes, or recipient names
   - Balance: Extract the closing balance for each transaction

3. Special Considerations:
   - Ignore any lines that are not actual transactions (e.g., headers, footers)
   - Ensure all numerical values are properly converted to floats
   - Maintain the chronological order of transactions

Parse the statement meticulously, ensuring all transactions are captured accurately. The output should strictly adhere to the provided JSON structure and parsing rules.

"""

def extract_pdf_text(pdf_files: List) -> str:
    """Extract text from uploaded PDF files using Marker"""
    text = ""
    converter = PdfConverter(artifact_dict=create_model_dict(),)
    for uploaded_file in pdf_files:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        rendered = converter("temp.pdf")
        extracted_text, _, _ = text_from_rendered(rendered)
        text += extracted_text 
    os.remove("temp.pdf")
    return text

def process_bank_transactions(text: str) -> json:
    """Process text through Ollama phi4 model"""
    ollama_host = os.getenv('OLLAMA_HOST', 'host.docker.internal')
    url = f"http://{ollama_host}:11434/api/chat"
    
    payload = {
        "model": "phi4",
        "messages": [
            {
                "role": "system",
                "content": f"{SYSTEM_MESSAGE}"
            },
            {
                "role": "user",
                "content": f"{text}"
            }
        ],
        "format": BankStatement.model_json_schema(),
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        response_data = response.json()
        
        if 'message' in response_data:
            try:
                content = response_data['message']['content']
                transactions = json.loads(content)
                return {"data": transactions}
            except json.JSONDecodeError:
                return {"error": "Failed to parse model response as JSON"}
        else:
            return {"error": "Unexpected response format from model"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out connecting to Ollama"}
    except requests.exceptions.ConnectionError:
        return {"error": "Failed to connect to Ollama server"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Streamlit Interface
st.title("Bank Transaction PDF Parser")

uploaded_files = st.file_uploader(
    "Upload Bank Statement PDFs",
    accept_multiple_files=True,
    type=['pdf']
)

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        extracted_text = extract_pdf_text(uploaded_files)
        result = process_bank_transactions(extracted_text)

        # Show extracted text
        st.subheader("Extracted Text")
        # with st.expander("View Extracted Text", expanded=True):
        st.text_area("", extracted_text, height=200)

        # Show API response
        st.subheader("API Response")
        with st.expander("View API Response", expanded=True):
            st.json(result)

        if "error" in result:
            st.error(f"Error processing transactions: {result['error']}")
        else:
            st.success("Successfully processed transactions")
            if "data" in result and result["data"]:
                try:
                    df = pd.DataFrame(result["data"]["transactions"])
                    st.subheader("Transactions")
                    with st.expander("View Transactions", expanded=True):
                        st.dataframe(df)
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(result, indent=2),
                        file_name="transactions.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Error creating dataframe: {str(e)}")
            else:
                st.warning("No transactions were extracted from the document")