import pandas as pd
import os
from openai import OpenAI
from typing import List, Dict
import json
from collections import defaultdict

# Initialize the OpenAI client
client = OpenAI(api_key="")

def load_comments():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the sentiment-analyzed comments
    input_file = os.path.join(current_dir, "COMMENTS", "video2_with_sentiment.csv")
    
    if not os.path.exists(input_file):
        print(f"Comments file not found at: {input_file}")
        exit()
    
    return pd.read_csv(input_file)

def get_topic_analysis_prompt(comments: List[str]) -> str:
    return f"""Analyze the following comments and group them into topics. For each topic:
1. Identify the main theme or subject
2. List 3-5 keywords that represent the topic
3. Provide a representative comment from the group

Comments to analyze:
{json.dumps(comments, indent=2)}

Please respond in the following JSON format:
{{
    "topics": [
        {{
            "topic_id": 1,
            "theme": "topic description",
            "keywords": ["keyword1", "keyword2", "keyword3"],
            "representative_comment": "example comment"
        }}
    ]
}}"""

def analyze_topics_with_gpt(comments_df, sentiment):
    # Filter comments by sentiment
    filtered_comments = comments_df[comments_df['Sentiment'] == sentiment]['Comment'].tolist()
    
    if not filtered_comments:
        print(f"No {sentiment} comments found.")
        return
    
    print(f"\nAnalyzing {sentiment} comments...")
    print(f"Number of {sentiment} comments: {len(filtered_comments)}")
    
    # If there are fewer than 2 comments, just show them directly
    if len(filtered_comments) < 2:
        print("\nComments:")
        for i, comment in enumerate(filtered_comments, 1):
            print(f"\nComment {i}:")
            print(comment)
        return
    
    print(f"\nAnalyzing topics for {sentiment} comments using GPT-3.5 Turbo...")
    
    try:
        # Prepare the prompt
        prompt = get_topic_analysis_prompt(filtered_comments)
        
        # Call GPT-3.5 Turbo using the new client syntax
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes comments and groups them into meaningful topics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent results
            max_tokens=1000
        )
        
        # Parse the response using the new response structure
        analysis = json.loads(response.choices[0].message.content)
        
        # Print results
        print(f"\nFound {len(analysis['topics'])} topics in {sentiment} comments:")
        print("\nTopic Analysis:")
        
        for topic in analysis['topics']:
            print(f"\nTopic {topic['topic_id']}:")
            print(f"Theme: {topic['theme']}")
            print(f"Keywords: {', '.join(topic['keywords'])}")
            print(f"Example comment: {topic['representative_comment'][:200]}...")
            
    except Exception as e:
        print(f"Error during GPT analysis: {str(e)}")
        return

def main():
    # Load comments
    print("Loading comments...")
    comments_df = load_comments()
    
    # Print overall sentiment distribution
    sentiment_counts = comments_df['Sentiment'].value_counts()
    print("\nOverall Sentiment Distribution:")
    print(sentiment_counts)
    
    # Analyze topics for positive comments
    analyze_topics_with_gpt(comments_df, "POSITIVE")
    
    # Analyze topics for negative comments
    analyze_topics_with_gpt(comments_df, "NEGATIVE")

if __name__ == "__main__":
    main() 