"""
Face Attendance System - Streamlit Web Application
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime, date
import time
import sys
import os
import sqlite3
import pytz

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer
from src.database import AttendanceDatabase
from src.liveness_detection import LivenessDetector
from src.attendance_system import AttendanceSystem

# Page configuration
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'system' not in st.session_state:
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'attendance.db')
    st.session_state.system = AttendanceSystem(DB_PATH)
    st.session_state.db = AttendanceDatabase(DB_PATH)

# Initialize settings in session state
if 'liveness_enabled' not in st.session_state:
    st.session_state.liveness_enabled = True

if 'recognition_threshold' not in st.session_state:
    st.session_state.recognition_threshold = 0.7

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "👤 Register User", "✅ Mark Attendance", "📊 View Records", "📈 Reports", "⚙️ Settings"]
)

# Helper function to get current datetime in IST
def get_current_datetime():
    """Get current datetime in IST timezone"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

# Helper function to capture image from webcam
def capture_image():
    """Capture image from webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return None, "Could not open webcam"
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None, "Failed to capture image"
    
    return frame, "Success"

# Helper function to display image
def display_image_with_box(frame, boxes=None, text=None):
    """Display image with optional bounding box and text"""
    display_frame = frame.copy()
    
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        cv2.rectangle(display_frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0), 3)
        
        if text:
            cv2.putText(display_frame, text,
                       (int(box[0]), int(box[1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Convert BGR to RGB
    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    return display_frame

# Helper function to safely format confidence
def format_confidence(x):
    """Safely format confidence score"""
    if x is None:
        return 'N/A'
    try:
        if isinstance(x, bytes):
            return 'Error'
        return f"{float(x):.4f}"
    except (ValueError, TypeError):
        return 'N/A'

#######################
# HOME PAGE
#######################
if page == "🏠 Home":
    st.markdown('<p class="main-header">👤 Face Attendance System</p>', unsafe_allow_html=True)
    
    # Display current date and time
    current_dt = get_current_datetime()
    st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>📅 {current_dt.strftime('%A, %B %d, %Y')} | 🕐 {current_dt.strftime('%I:%M:%S %p')}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👥 Total Users")
        users = st.session_state.db.get_all_users()
        st.markdown(f"<h1 style='text-align: center; color: #1f77b4;'>{len(users)}</h1>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ✅ Today's Attendance")
        today_date = current_dt.date()
        today_records = st.session_state.db.get_attendance_records(date=today_date)
        unique_users = len(set(r['employee_id'] for r in today_records))
        st.markdown(f"<h1 style='text-align: center; color: #2ca02c;'>{unique_users}</h1>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("### 📊 Total Records")
        all_records = st.session_state.db.get_attendance_records(limit=10000)
        st.markdown(f"<h1 style='text-align: center; color: #ff7f0e;'>{len(all_records)}</h1>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System features
    st.markdown("### 🎯 System Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ✅ **Face Detection & Recognition**
        - MTCNN for accurate face detection
        - FaceNet for face recognition
        - 512-dimensional embeddings
        
        ✅ **User Management**
        - Multi-image registration
        - Employee ID tracking
        - Department organization
        """)
    
    with col2:
        st.markdown("""
        ✅ **Attendance Features**
        - Automatic punch-in/out detection
        - Liveness detection (anti-spoofing)
        - Duplicate prevention
        
        ✅ **Reporting & Analytics**
        - Daily attendance reports
        - Work hours calculation
        - Export to CSV
        """)
    
    st.markdown("---")
    
    # Recent activity
    st.markdown("### 📌 Recent Activity")
    recent_records = st.session_state.db.get_attendance_records(limit=5)
    
    if recent_records:
        df_recent = pd.DataFrame(recent_records)
        df_recent = df_recent[['name', 'employee_id', 'punch_type', 'timestamp', 'confidence_score']]
        df_recent.columns = ['Name', 'Employee ID', 'Punch Type', 'Timestamp', 'Confidence']
        df_recent['Confidence'] = df_recent['Confidence'].apply(format_confidence)
        st.dataframe(df_recent, use_container_width=True)
    else:
        st.info("No attendance records yet")

#######################
# REGISTER USER PAGE
#######################
elif page == "👤 Register User":
    st.markdown('<p class="main-header">👤 Register New User</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 User Information")
        
        name = st.text_input("Full Name*", placeholder="Enter full name")
        employee_id = st.text_input("Employee ID*", placeholder="e.g., EMP001")
        department = st.selectbox(
            "Department*",
            ["Engineering", "HR", "Finance", "Marketing", "Operations", "IT", "Admin", "Other"]
        )
        
        num_images = st.slider("Number of images to capture", 3, 10, 5)
        delay = st.slider("Delay between captures (seconds)", 1, 5, 2)
        
        st.markdown("---")
        
        register_button = st.button("📸 Start Registration", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Registration Instructions")
        st.markdown("""
        1. **Fill in user details** on the left
        2. **Click 'Start Registration'** button
        3. **Position your face** clearly in camera view
        4. **Look at different angles** for each capture:
           - Straight ahead
           - Slightly left
           - Slightly right
           - Slightly up
           - Slightly down
        5. **Keep good lighting** and avoid shadows
        6. **Remove glasses/masks** if possible
        
        ⚠️ **Important Notes:**
        - Keep your face centered in the frame
        - Avoid excessive movement during capture
        - Ensure good lighting conditions
        - System will capture multiple images automatically
        """)
    
    if register_button:
        if not name or not employee_id:
            st.error("❌ Please fill in all required fields (Name and Employee ID)")
        else:
            with st.spinner("🔄 Starting registration process..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                image_placeholder = st.empty()
                
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not open webcam")
                else:
                    captured_faces = []
                    captured_frames = []
                    
                    for i in range(num_images):
                        status_text.markdown(f"📸 Capturing image {i+1}/{num_images}...")
                        progress_bar.progress((i + 1) / num_images)
                        
                        time.sleep(delay)
                        
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        # Detect face
                        boxes, probs = st.session_state.system.detector.detect_face(frame)
                        
                        # Display frame
                        if boxes is not None and len(boxes) > 0:
                            display_frame = display_image_with_box(
                                frame, boxes, 
                                f"Captured {i+1}/{num_images} - {probs[0]:.3f}"
                            )
                            image_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                            
                            # Extract face
                            face_tensor = st.session_state.system.detector.extract_face(frame)
                            if face_tensor is not None:
                                captured_faces.append(face_tensor)
                                captured_frames.append(frame)
                        else:
                            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                    
                    cap.release()
                    
                    if len(captured_faces) < 3:
                        st.error(f"❌ Registration failed: Only captured {len(captured_faces)} valid images. Need at least 3.")
                    else:
                        # Extract embeddings
                        status_text.markdown("🔄 Processing face embeddings...")
                        embeddings = st.session_state.system.recognizer.extract_embeddings_batch(captured_faces)
                        avg_embedding = st.session_state.system.recognizer.average_embeddings(embeddings)
                        
                        # Save to database
                        status_text.markdown("💾 Saving to database...")
                        user_id = st.session_state.db.add_user(
                            name, employee_id, department, avg_embedding, len(captured_faces)
                        )
                        
                        if user_id:
                            st.markdown(f'<div class="success-box">✅ <b>Registration Successful!</b><br>User ID: {user_id}<br>Name: {name}<br>Employee ID: {employee_id}<br>Images captured: {len(captured_faces)}</div>', unsafe_allow_html=True)
                            
                            # Display captured images
                            st.markdown("### 📷 Captured Images")
                            cols = st.columns(min(len(captured_frames), 5))
                            for idx, frame in enumerate(captured_frames[:5]):
                                with cols[idx]:
                                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Image {idx+1}")
                        else:
                            st.error("❌ Registration failed: User with this name or employee ID already exists")
    
    # View registered users
    st.markdown("---")
    st.markdown("### 👥 Registered Users")
    
    users = st.session_state.db.get_all_users()
    
    if users:
        df_users = pd.DataFrame([{
            'User ID': u['user_id'],
            'Name': u['name'],
            'Employee ID': u['employee_id'],
            'Department': u['department'],
            'Images': u['num_images'],
            'Registered': u['created_at']
        } for u in users])
        
        st.dataframe(df_users, use_container_width=True)
        
        # Delete user
        st.markdown("### 🗑️ Delete User")
        user_to_delete = st.selectbox("Select user to delete", [f"{u['user_id']} - {u['name']}" for u in users])
        
        if st.button("Delete User", type="secondary"):
            user_id = int(user_to_delete.split(" - ")[0])
            st.session_state.db.delete_user(user_id)
            st.success(f"✅ User {user_id} deleted successfully")
            st.rerun()
    else:
        st.info("No users registered yet")

#######################
# MARK ATTENDANCE PAGE
#######################
elif page == "✅ Mark Attendance":
    st.markdown('<p class="main-header">✅ Mark Attendance</p>', unsafe_allow_html=True)
    
    # Display current settings
    current_dt = get_current_datetime()
    st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>📅 {current_dt.strftime('%A, %B %d, %Y')} | 🕐 {current_dt.strftime('%I:%M:%S %p')}</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### ⚙️ Current Settings")
        
        st.markdown(f"""
        **Active Configuration:**
        - Liveness Detection: {'✅ Enabled' if st.session_state.liveness_enabled else '❌ Disabled'}
        - Recognition Threshold: {st.session_state.recognition_threshold:.2f}
        
        ℹ️ To change settings, go to **⚙️ Settings** page
        """)
        
        st.markdown("---")
        
        mark_button = st.button("📸 Mark Attendance", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. **Check current settings** on the left
        2. **Click 'Mark Attendance'** button
        3. If liveness enabled:
           - Move your head slightly
           - System will detect natural motion
        4. **Look at the camera** for face capture
        5. System will recognize you and mark attendance
        
        ⚠️ **Notes:**
        - Ensure good lighting
        - Face the camera directly
        - Remove obstructions (glasses, masks if possible)
        - System prevents duplicate punches within 1 minute
        - Adjust settings in ⚙️ Settings page if needed
        """)
    
    if mark_button:
        with st.spinner("🔄 Processing attendance..."):
            result_placeholder = st.empty()
            image_placeholder = st.empty()
            
            # Liveness check
            if st.session_state.liveness_enabled:
                result_placeholder.info("🔄 Performing liveness detection... Please move your head slightly")
                is_live, motion = st.session_state.system.liveness.detect_motion()
                
                if not is_live:
                    result_placeholder.markdown('<div class="error-box">❌ <b>Liveness Check Failed</b><br>No natural motion detected. Please try again.</div>', unsafe_allow_html=True)
                    st.stop()
                else:
                    result_placeholder.success(f"✅ Liveness check passed (motion: {motion:.2f})")
                    time.sleep(1)
            
            # Capture face
            result_placeholder.info("📸 Capturing face...")
            frame, msg = capture_image()
            
            if frame is None:
                result_placeholder.error(f"❌ {msg}")
                st.stop()
            
            # Detect face
            boxes, probs = st.session_state.system.detector.detect_face(frame)
            
            if boxes is None or len(boxes) == 0:
                result_placeholder.markdown('<div class="error-box">❌ <b>No Face Detected</b><br>Please ensure your face is clearly visible and try again.</div>', unsafe_allow_html=True)
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                st.stop()
            
            # Extract embedding
            result_placeholder.info("🔄 Recognizing face...")
            face_tensor = st.session_state.system.detector.extract_face(frame)
            
            if face_tensor is None:
                result_placeholder.error("❌ Failed to extract face")
                st.stop()
            
            embedding = st.session_state.system.recognizer.extract_embedding(face_tensor)
            
            # Match with database
            matched_user = st.session_state.db.find_matching_user(embedding, st.session_state.recognition_threshold)
            
            if matched_user is None:
                result_placeholder.markdown('<div class="error-box">❌ <b>User Not Recognized</b><br>Similarity score below threshold. Please register first or adjust threshold in Settings.</div>', unsafe_allow_html=True)
                display_frame = display_image_with_box(frame, boxes, "Unknown")
                image_placeholder.image(display_frame, channels="RGB", use_container_width=True)
                st.stop()
            
            # Determine punch type
            last_attendance = st.session_state.db.get_last_attendance(matched_user['user_id'])
            
            if last_attendance is None:
                punch_type = 'IN'
            else:
                last_time = datetime.strptime(last_attendance['timestamp'], '%Y-%m-%d %H:%M:%S')
                time_diff = get_current_datetime().replace(tzinfo=None) - last_time
                
                if time_diff.total_seconds() < 60:
                    result_placeholder.markdown(f'<div class="error-box">⚠️ <b>Duplicate Punch Attempt</b><br>Last punch was {int(time_diff.total_seconds())} seconds ago. Please wait at least 1 minute.</div>', unsafe_allow_html=True)
                    st.stop()
                
                punch_type = 'OUT' if last_attendance['punch_type'] == 'IN' else 'IN'
            
            # Mark attendance
            attendance_id = st.session_state.db.mark_attendance(
                matched_user['user_id'],
                punch_type,
                matched_user['similarity']
            )
            
            current_time = get_current_datetime()
            
            # Display result
            color = "🟢" if punch_type == 'IN' else "🔴"
            result_placeholder.markdown(f'''
            <div class="success-box">
                <h3>{color} Attendance Marked Successfully!</h3>
                <p><b>Attendance ID:</b> {attendance_id}</p>
                <p><b>Name:</b> {matched_user['name']}</p>
                <p><b>Employee ID:</b> {matched_user['employee_id']}</p>
                <p><b>Department:</b> {matched_user['department']}</p>
                <p><b>Action:</b> PUNCH-{punch_type}</p>
                <p><b>Date:</b> {current_time.strftime('%Y-%m-%d')}</p>
                <p><b>Time:</b> {current_time.strftime('%I:%M:%S %p')}</p>
                <p><b>Confidence:</b> {matched_user['similarity']:.4f}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            display_frame = display_image_with_box(
                frame, boxes, 
                f"{matched_user['name']} - PUNCH {punch_type}"
            )
            image_placeholder.image(display_frame, channels="RGB", use_container_width=True)

#######################
# VIEW RECORDS PAGE
#######################
elif page == "📊 View Records":
    st.markdown('<p class="main-header">📊 View Attendance Records</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        users = st.session_state.db.get_all_users()
        user_options = ["All Users"] + [f"{u['user_id']} - {u['name']}" for u in users]
        selected_user = st.selectbox("Filter by User", user_options)
    
    with col2:
        date_filter = st.date_input("Filter by Date", value=None)
    
    with col3:
        limit = st.number_input("Number of records", min_value=10, max_value=1000, value=50, step=10)
    
    # Get records
    user_id = None if selected_user == "All Users" else int(selected_user.split(" - ")[0])
    date_str = date_filter.strftime('%Y-%m-%d') if date_filter else None
    
    records = st.session_state.db.get_attendance_records(user_id=user_id, date=date_str, limit=limit)
    
    if records:
        df_records = pd.DataFrame(records)
        df_records = df_records[['attendance_id', 'name', 'employee_id', 'department', 'punch_type', 'timestamp', 'confidence_score']]
        df_records.columns = ['ID', 'Name', 'Employee ID', 'Department', 'Punch Type', 'Timestamp', 'Confidence']
        df_records['Confidence'] = df_records['Confidence'].apply(format_confidence)
        
        st.dataframe(df_records, use_container_width=True)
        
        # Download button
        csv = df_records.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"attendance_records_{get_current_datetime().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No records found with selected filters")

#######################
# REPORTS PAGE
#######################
elif page == "📈 Reports":
    st.markdown('<p class="main-header">📈 Daily Attendance Report</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    report_date = st.date_input("Select Date", value=get_current_datetime().date())
    
    if st.button("Generate Report", type="primary"):
        date_str = report_date.strftime('%Y-%m-%d')
        report_data = st.session_state.db.get_daily_report(date_str)
        
        if report_data:
            df_report = pd.DataFrame(report_data)
            df_report = df_report[['name', 'employee_id', 'department', 'punch_in', 'punch_out', 'total_hours', 'status']]
            df_report.columns = ['Name', 'Employee ID', 'Department', 'Punch In', 'Punch Out', 'Hours', 'Status']
            
            # Format columns
            df_report['Punch In'] = df_report['Punch In'].apply(lambda x: x.split()[1] if x else 'N/A')
            df_report['Punch Out'] = df_report['Punch Out'].apply(lambda x: x.split()[1] if x else 'N/A')
            df_report['Hours'] = df_report['Hours'].apply(lambda x: f"{x:.2f}" if x > 0 else 'N/A')
            
            st.dataframe(df_report, use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.markdown("### 📊 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Employees", len(report_data))
            
            with col2:
                present = sum(1 for r in report_data if r['status'] in ['COMPLETE', 'ACTIVE'])
                st.metric("Present", present)
            
            with col3:
                complete = sum(1 for r in report_data if r['status'] == 'COMPLETE')
                st.metric("Complete Sessions", complete)
            
            with col4:
                active = sum(1 for r in report_data if r['status'] == 'ACTIVE')
                st.metric("Active Sessions", active)
            
            # Download button
            csv = df_report.to_csv(index=False)
            st.download_button(
                label="📥 Download Report",
                data=csv,
                file_name=f"daily_report_{date_str}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No attendance records for {date_str}")

#######################
# SETTINGS PAGE
#######################
elif page == "⚙️ Settings":
    st.markdown('<p class="main-header">⚙️ System Settings</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Attendance Settings
    st.markdown("### 🎛️ Attendance Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Liveness Detection")
        liveness_enabled = st.checkbox(
            "Enable Liveness Detection",
            value=st.session_state.liveness_enabled,
            help="Detect natural motion to prevent spoofing attacks"
        )
        
        if liveness_enabled != st.session_state.liveness_enabled:
            st.session_state.liveness_enabled = liveness_enabled
            st.success(f"✅ Liveness detection {'enabled' if liveness_enabled else 'disabled'}")
    
    with col2:
        st.markdown("#### Recognition Threshold")
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.5,
            max_value=0.9,
            value=st.session_state.recognition_threshold,
            step=0.05,
            help="Higher = stricter matching, Lower = more lenient"
        )
        
        if threshold != st.session_state.recognition_threshold:
            st.session_state.recognition_threshold = threshold
            st.success(f"✅ Threshold set to {threshold:.2f}")
    
    st.markdown("---")
    
    st.markdown(f"""
    **Current Configuration:**
    - Liveness Detection: {'✅ Enabled' if st.session_state.liveness_enabled else '❌ Disabled'}
    - Recognition Threshold: {st.session_state.recognition_threshold:.2f}
    
    ℹ️ **Threshold Guide:**
    - 0.5-0.6: Very lenient (may have false positives)
    - 0.65-0.75: Balanced (recommended)
    - 0.75-0.9: Very strict (may have false negatives)
    """)
    
    st.markdown("---")
    
    # System Information
    st.markdown("### 🔧 System Information")
    
    import torch
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Device Information:**
        - Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
        - PyTorch Version: {torch.__version__}
        """)
        
        if torch.cuda.is_available():
            st.markdown(f"""
            - GPU: {torch.cuda.get_device_name(0)}
            - CUDA Version: {torch.version.cuda}
            - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB
            """)
    
    with col2:
        users = st.session_state.db.get_all_users()
        all_records = st.session_state.db.get_attendance_records(limit=10000)
        current_dt = get_current_datetime()
        
        st.markdown(f"""
        **Database Statistics:**
        - Total Users: {len(users)}
        - Total Attendance Records: {len(all_records)}
        - Current Date: {current_dt.strftime('%Y-%m-%d')}
        - Current Time: {current_dt.strftime('%I:%M:%S %p')}
        - Database Location: `{st.session_state.db.db_path}`
        """)
    
    st.markdown("---")
    
    # Database Maintenance
    st.markdown("### 🗑️ Database Maintenance")
    
    st.warning("⚠️ **Caution:** This will delete all attendance records. User registrations will NOT be affected.")
    
    if st.button("🧹 Clean All Attendance Records", type="secondary"):
        conn = sqlite3.connect(st.session_state.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM attendance")
        count_before = cursor.fetchone()[0]
        
        if count_before > 0:
            confirm_col1, confirm_col2 = st.columns([1, 3])
            with confirm_col1:
                if st.button("⚠️ CONFIRM DELETE"):
                    cursor.execute("DELETE FROM attendance")
                    conn.commit()
                    conn.close()
                    
                    st.success(f"✅ Deleted {count_before} attendance records successfully!")
                    st.info("Database is now clean. You can mark fresh attendance.")
                    time.sleep(2)
                    st.rerun()
        else:
            st.info("No attendance records to clean")
        
        conn.close()
    
    st.markdown("---")
    
    st.markdown("### 🎨 About")
    st.markdown(f"""
    **Face Attendance System**  
    Version 1.0  
    Current Date & Time: {get_current_datetime().strftime('%Y-%m-%d %I:%M:%S %p IST')}
    
    Built with:
    - FaceNet (InceptionResnetV1)
    - MTCNN Face Detector
    - PyTorch
    - Streamlit
    
    **Features:**
    - Face detection and recognition
    - Liveness detection (motion-based)
    - Automatic punch-in/out
    - Daily reports
    - Work hours calculation
    - IST timezone support
    
    **Approach 1:** Pre-trained FaceNet + Transfer Learning
    - Using pre-trained FaceNet model on VGGFace2
    - 512-dimensional embeddings
    - Cosine similarity for matching
    - MTCNN for face detection and alignment
    """)