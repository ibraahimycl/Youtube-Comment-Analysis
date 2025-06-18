from get_id import extract_channel_id, get_uploads_playlist_id, get_videos_by_date_range, extract_playlist_id, get_video_title
from comment import video_comments
import pandas as pd
import os
from datetime import datetime
import re

API_KEY = ""

def sanitize_filename(title):
    # Remove special characters and replace spaces with underscores
    sanitized = re.sub(r'[^\w\s-]', '', title.lower())
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized

def process_video_comments():
    # Get input type from user
    print("\nChoose input type:")
    print("1. Channel URL")
    print("2. Playlist URL")
    choice = input("Enter your choice (1 or 2): ")
    
    # Get date range from user
    start_date_str = input("Enter start date (YYYY-MM-DD): ")
    end_date_str = input("Enter end date (YYYY-MM-DD): ")
    
    # Convert dates to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
    
    # Get video IDs based on input type
    print("\nGetting video information...")
    if choice == "1":
        channel_url = input("Enter the channel URL: ")
        channel_id = extract_channel_id(channel_url, API_KEY)
        if not channel_id:
            print("Could not get channel ID.")
            return
        
        playlist_id = get_uploads_playlist_id(channel_id)
        video_ids = get_videos_by_date_range(playlist_id, start_date, end_date, is_channel_uploads=True)
    else:
        playlist_url = input("Enter the playlist URL: ")
        playlist_id = extract_playlist_id(playlist_url, API_KEY)
        if not playlist_id:
            print("Could not get playlist ID.")
            return
        
        video_ids = get_videos_by_date_range(playlist_id, start_date, end_date)
    
    if not video_ids:
        print("No videos found in the specified date range.")
        return
    
    print(f"\nFound {len(video_ids)} videos. Processing comments...")
    
    # Create a directory for the CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"comments_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # List to store all CSV filenames
    csv_files = []
    
    # Process each video
    for i, video_id in enumerate(video_ids, 1):
        print(f"\nProcessing video {i}/{len(video_ids)} (ID: {video_id})")
        
        # Get video title
        video_title = get_video_title(video_id, API_KEY)
        if video_title:
            # Sanitize the title for filename
            safe_title = sanitize_filename(video_title)
            csv_filename = os.path.join(output_dir, f"{safe_title}.csv")
        else:
            # Fallback to video ID if title can't be retrieved
            csv_filename = os.path.join(output_dir, f"video_{video_id}.csv")
        
        # Get comments and save to CSV
        try:
            video_comments(video_id, csv_filename)
            csv_files.append(csv_filename)
        except Exception as e:
            print(f"Error getting comments for video {video_id}: {e}")
    
    if not csv_files:
        print("Could not get comments for any videos.")
        return
    
    # Merge all CSV files
    print("\nMerging all comment files...")
    all_comments = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add video title column
            video_title = os.path.basename(csv_file).replace('.csv', '')
            df['Video Title'] = video_title
            all_comments.append(df)
        except Exception as e:
            print(f"Error reading file {csv_file}: {e}")
    
    if all_comments:
        # Combine all dataframes
        merged_df = pd.concat(all_comments, ignore_index=True)
        
        # Save merged file
        merged_filename = os.path.join(output_dir, "all_comments.csv")
        merged_df.to_csv(merged_filename, index=False)
        print(f"\nAll comments merged: {merged_filename}")
    else:
        print("No comments to merge.")

if __name__ == "__main__":
    process_video_comments() 