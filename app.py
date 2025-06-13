import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import face_recognition
import cv2
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import math

app = Flask(__name__)
app.secret_key = "db50079661ce9428cf09d437a3749baa"
UPLOAD_FOLDER = './uploads'

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Configuration Flags ---
# Set to True to use live webcam, False to use a recorded video file
# IMPORTANT: For GitHub Codespaces, IS_LIVE should generally be False as webcam access is not direct.
IS_LIVE = False
# Set to True to send email notifications, False to disable
SEND_MAIL = False
# --- End Configuration Flags ---

# Email configuration
SENDER_EMAIL = ""  # Your email address
RECEIVER_EMAIL = ""  # Receiver's email address
SMTP_SERVER = "smtp.gmail.com"  # SMTP server address (e.g., Gmail: smtp.gmail.com)
SMTP_PORT = 587  # Port for sending email (587 for TLS)
SENDER_PASSWORD = ""   # Your email app-specific password

# --- Video File Path (ONLY relevant if IS_LIVE is False) ---
# IMPORTANT: Update this path to your recorded video file.
# Make sure 'test_video.mp4' is present in your 'uploads' folder for testing.
RECORDED_VIDEO_PATH = os.path.join(UPLOAD_FOLDER, 'test_fail.mp4')
# --- End Video File Path ---

# Function to check if the file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to extract images from a PDF
def extract_images_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    return images

# Function to send an email when a match is found
def send_email(location, email_subject="Face Match Detected!", email_body_prefix="A match has been found"):
    if not SEND_MAIL:
        print("Email sending is disabled by SEND_MAIL flag.")
        return

    try:
        # Set up the MIME
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = email_subject

        # Body of the email
        body = f"{email_body_prefix} with the uploaded image. Location: {location}"
        msg.attach(MIMEText(body, 'plain'))

        # Set up the server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        
        # Send the email
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"Email sent successfully to {RECEIVER_EMAIL}!")

    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Function to detect faces using video source and match with uploaded image
def detect_faces_from_video_source(uploaded_image_path):
    try:
        # Load the uploaded image and extract its face encoding
        uploaded_image = face_recognition.load_image_file(uploaded_image_path)
        uploaded_encoding = face_recognition.face_encodings(uploaded_image)[0]

        detection_timestamp = None # Will store the time of detection
        match_image_filename = None

        # Initialize the video capture based on IS_LIVE flag
        if IS_LIVE:
            cap = cv2.VideoCapture(0)  # Use live webcam
            source_description = "webcam"
        else:
            if not os.path.exists(RECORDED_VIDEO_PATH):
                return {
                    "message": f"Error: Recorded video file not found at {RECORDED_VIDEO_PATH}. Please check the path.",
                    "detection_time": None,
                    "match_image_url": None
                }
            cap = cv2.VideoCapture(RECORDED_VIDEO_PATH) # Use recorded video
            source_description = f"video file ({os.path.basename(RECORDED_VIDEO_PATH)})"
            
            # Get video properties for timestamp calculation
            fps = cap.get(cv2.CAP_PROP_FPS)


        if not cap.isOpened():
            return {
                "message": f"Could not access the {source_description}. Please check the source.",
                "detection_time": None,
                "match_image_url": None
            }

        print(f"Starting face detection from {source_description}...")
        
        frame_number = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"End of {source_description} or failed to read frame.")
                break # Break if no frame is read (end of video or webcam error)

            # Increment frame number
            frame_number += 1

            # Convert the frame to RGB (OpenCV uses BGR by default)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the current frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Compare detected faces with the uploaded face encoding
            for encoding in face_encodings:
                matches = face_recognition.compare_faces([uploaded_encoding], encoding)
                if True in matches:
                    cap.release()
                    location = face_locations[0]  # Get the face location (top, right, bottom, left)

                    # Calculate detection timestamp ONLY if it's a recorded video
                    if not IS_LIVE and fps > 0:
                        seconds_at_detection = frame_number / fps
                        minutes = math.floor(seconds_at_detection / 60)
                        seconds = math.floor(seconds_at_detection % 60)
                        milliseconds = math.floor((seconds_at_detection - math.floor(seconds_at_detection)) * 1000)
                        detection_timestamp = f"{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"
                        print(f"Face detected at: {detection_timestamp}")


                    # Draw green bounding box on the original frame (BGR format)
                    top, right, bottom, left = location
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2) # Green color, 2px thickness

                    # Save the annotated image frame ONLY if it's a recorded video
                    if not IS_LIVE:
                        match_image_filename = "detected_face_frame.jpg"
                        match_image_path_full = os.path.join(app.config['UPLOAD_FOLDER'], match_image_filename)
                        cv2.imwrite(match_image_path_full, frame)
                        print(f"Detected frame saved to: {match_image_path_full}")
                    
                    send_email(location) # Send email with face location
                    
                    return {
                        "message": f"Match found in {source_description}! Location: {location}",
                        "detection_time": detection_timestamp, # Pass the detection timestamp
                        "match_image_url": url_for('uploaded_file', filename=match_image_filename) if match_image_filename else None
                    }

            # Display the current frame (only if running in an environment with display support)
            # This line will likely cause issues in Codespaces if you don't have a VNC/X11 setup.
            # You might want to comment it out or wrap it in a condition for Codespaces.
            # if IS_LIVE: # Only show live feed if IS_LIVE is True and display is available
            #     cv2.imshow(f"Face Detection ({source_description})", frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

        cap.release()
        # cv2.destroyAllWindows() # Only needed if cv2.imshow was used
        return {
            "message": f"No match found in {source_description}.",
            "detection_time": None,
            "match_image_url": None
        }

    except Exception as e:
        return {
            "message": f"An error occurred during face detection: {str(e)}",
            "detection_time": None,
            "match_image_url": None
        }

@app.route('/')
def home():
    """
    Render the home page with the upload form.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handles file uploads and initiates face detection.
    """
    if 'file' not in request.files:
        flash('Please upload an image.')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No file selected!')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Only image files are allowed!')
        return redirect(request.url)

    # Save the uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Check if the file is a PDF and extract the first page as an image
    if filename.endswith('.pdf'):
        try:
            images = convert_from_path(filepath)
            if not images:
                flash('No images could be extracted from the PDF.')
                return redirect(request.url)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted_image.jpg')
            images[0].save(image_path, 'JPEG')
            filepath = image_path  # Update filepath to the extracted image
        except Exception as e:
            flash(f"Error extracting image from PDF: {str(e)}")
            return redirect(request.url)

    # Detect faces using the chosen video source and compare with uploaded image
    detection_result = detect_faces_from_video_source(filepath)
    
    # Pass all relevant details to the template
    return render_template('result.html', 
                           result_message=detection_result['message'],
                           detection_time=detection_result['detection_time'], # Pass detection timestamp
                           match_image_url=detection_result['match_image_url'])

# Route to serve uploaded files (like the detected frame image)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)