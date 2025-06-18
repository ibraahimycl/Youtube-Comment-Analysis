import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def analyze_sentiment(comments_df):
    # Load the model and tokenizer
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define sentiment labels
    sentiment_labels = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    
    # Function to get sentiment for a single comment
    def get_sentiment(comment):
        if not isinstance(comment, str):
            return "NEUTRAL"
            
        # Tokenize and prepare input
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(scores, dim=1).item()
            
        return sentiment_labels[predicted_class]
    
    # Apply sentiment analysis to all comments
    print("Performing sentiment analysis...")
    comments_df['Sentiment'] = comments_df['Comment'].apply(get_sentiment)
    
    return comments_df

def process_comments_file(input_file):
    # Read the CSV file
    print(f"Reading comments from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Perform sentiment analysis
    df_with_sentiment = analyze_sentiment(df)
    
    # Save the results
    output_file = input_file.replace('.csv', '_with_sentiment.csv')
    df_with_sentiment.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print sentiment distribution
    sentiment_counts = df_with_sentiment['Sentiment'].value_counts()
    print("\nSentiment Distribution:")
    print(sentiment_counts)

if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use the correct path to the comments file
    input_file = os.path.join(current_dir, "COMMENTS", "video2.csv")
    
    if not os.path.exists(input_file):
        print(f"Comments file not found at: {input_file}")
        exit()
    
    process_comments_file(input_file) 