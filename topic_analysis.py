import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os

def load_comments():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the sentiment-analyzed comments
    input_file = os.path.join(current_dir, "COMMENTS", "all_comments_with_sentiment.csv")
    
    if not os.path.exists(input_file):
        print(f"Comments file not found at: {input_file}")
        exit()
    
    return pd.read_csv(input_file)

def analyze_topics(comments_df, sentiment):
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
    
    print(f"\nAnalyzing topics for {sentiment} comments...")
    
    # Initialize BERTopic
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    topic_model = BERTopic(embedding_model=sentence_model, 
                          min_topic_size=2,  # Minimum number of comments per topic
                          nr_topics="auto")  # Automatically determine number of topics
    
    # Fit the model
    topics, probs = topic_model.fit_transform(filtered_comments)
    
    # Get topic information
    topic_info = topic_model.get_topic_info()
    
    # Print results
    print(f"\nFound {len(topic_info)} topics in {sentiment} comments:")
    print("\nTopic Analysis:")
    
    for topic_id in topic_info['Topic'].tolist():
        if topic_id == -1:  # Skip outlier topic
            continue
            
        # Get top words for the topic
        topic_words = topic_model.get_topic(topic_id)
        words = [word for word, _ in topic_words[:5]]  # Get top 5 words
        
        # Get example comments for this topic
        topic_comments = [comment for comment, topic in zip(filtered_comments, topics) if topic == topic_id]
        example_comment = topic_comments[0] if topic_comments else "No example available"
        
        print(f"\nTopic {topic_id}:")
        print(f"Keywords: {', '.join(words)}")
        print(f"Example comment: {example_comment[:100]}...")  # Show first 100 chars of example

def main():
    # Load comments
    print("Loading comments...")
    comments_df = load_comments()
    
    # Print overall sentiment distribution
    sentiment_counts = comments_df['Sentiment'].value_counts()
    print("\nOverall Sentiment Distribution:")
    print(sentiment_counts)
    
    # Analyze topics for positive comments
    analyze_topics(comments_df, "POSITIVE")
    
    # Analyze topics for negative comments
    analyze_topics(comments_df, "NEGATIVE")

if __name__ == "__main__":
    main() 