/* Chat Messages Background Image */
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    
    /* Add background image */
    background-image: url('YOUR_IMAGE_URL_HERE');
    background-size: cover;  /* or 'contain' or '100% 100%' */
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: local;  /* scrolls with content */
    
    /* Optional: Add overlay to make text more readable */
    position: relative;
}

/* Optional: Semi-transparent overlay for better text readability */
.chat-messages::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.85);  /* White overlay at 85% opacity */
    z-index: 0;
    pointer-events: none;
}

/* Make sure messages appear above the overlay */
.chat-message,
.typing-indicator,
.script-button-group,
.upload-status {
    position: relative;
    z-index: 1;
}

Options for the image URL:

External URL: https://example.com/image.jpg
Base64 encoded: data:image/jpeg;base64,/9j/4AAQ...
Relative path: ./images/chat-bg.jpg

Different background styles you can try:

/* Subtle pattern */
background-size: 400px 400px;
background-repeat: repeat;

/* Watermark style */
background-size: 30%;
background-position: center;
background-repeat: no-repeat;
opacity: 0.1;

/* Full cover with blur */
background-size: cover;
filter: blur(2px);

/* Gradient overlay on image */
background: 
    linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)),
    url('YOUR_IMAGE_URL');

/* Video background */
Method 1: HTML5 Video Background (Recommended)
Add this HTML right after the opening <div class="chat-messages" id="chat-messages"> tag:

<div class="chat-messages" id="chat-messages">
    <video autoplay muted loop playsinline class="chat-bg-video">
        <source src="YOUR_VIDEO_URL.mp4" type="video/mp4">
        <source src="YOUR_VIDEO_URL.webm" type="video/webm">
    </video>
    <!-- Messages will appear here -->
</div>
Then add this CSS:

/* Video Background */
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    position: relative;
}

.chat-bg-video {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    z-index: 0;
    pointer-events: none;
}

/* Optional: Add overlay for better text readability */
.chat-messages::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(255, 255, 255, 0.85);
    z-index: 1;
    pointer-events: none;
}

/* Make sure messages appear above video and overlay */
.chat-message,
.typing-indicator,
.script-button-group,
.upload-status {
    position: relative;
    z-index: 2;
}

Method 2: Background Video with CSS (External hosting)

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    position: relative;
    background: url('https://example.com/video-frame.jpg') center/cover;
}

.chat-messages::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.9);
    z-index: 0;
}

Video Hosting Options:

Your own server: Upload .mp4 and .webm files
YouTube/Vimeo embed: (not recommended for background, causes performance issues)
CDN services: Cloudinary, AWS S3, etc.

Important Tips:

Keep video file under 5MB for performance
Use short loops (5-15 seconds)
Set muted attribute (autoplay requires muted videos)
Use playsinline for mobile compatibility
Consider animated GIF as a lighter alternative

Example with popular video formats:

<video autoplay muted loop playsinline class="chat-bg-video">
    <source src="https://your-cdn.com/legal-bg.mp4" type="video/mp4">
    <source src="https://your-cdn.com/legal-bg.webm" type="video/webm">
    <!-- Fallback image -->
    <img src="fallback-image.jpg" alt="Background">
</video>