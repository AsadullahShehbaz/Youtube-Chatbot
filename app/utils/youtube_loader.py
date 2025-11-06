# app/utils/youtube_loader.py

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

def fetch_transcript(video_id: str) -> str:
    """
    Fetches the transcript text from a YouTube video.
    Args:
        video_id (str): YouTube video ID
    Returns:
        str: Combined transcript text
    """
    try:
        transcript_obj = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        # Combine all text snippets into one string
        transcript = " ".join(snippet.text for snippet in transcript_obj.snippets)
        return transcript
    except TranscriptsDisabled:
        raise Exception("No captions available for this video.")
    except Exception as e:
        raise Exception(f"Error fetching transcript: {e}")

fetch_transcript('67_aMPDk2zw')
print('done')