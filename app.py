from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import cv2
import json                                         
import os
from datetime import datetime, timedelta
from utils.face_utils import recognize_face_at_location, get_face_encodings, verify_liveness, detector, predictor
from utils.pdf_utils import generate_student_report_pdf
import numpy as np

# Try optional imports
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

from twilio.rest import Client
import pandas as pd
import io
import time
import threading
from sqlalchemy.exc import IntegrityError

# Helper to always get Indian Standard Time (+5:30)
def get_ist_time():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)

# Tracker for live faculty presence in classroom sessions
live_faculty_sessions = {}

app = Flask(__name__)

# Configuration from environment variables
app.config['SECRET_KEY'] = 'smart_attendance_secret_key'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:123456@localhost/smart_attendance'

# Set absolute path for uploads
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'static', 'uploads')

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

@app.context_processor
def inject_now():
    return {'get_ist_time': get_ist_time}

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Twilio Configuration from environment variables
TWILIO_ACCOUNT_SID = 'your_sid'
TWILIO_AUTH_TOKEN = 'your_token'
TWILIO_PHONE_NUMBER = 'your_phone'

# --- Models ---
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum('admin', 'faculty', 'student'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    face_encoding = db.Column(db.Text) # Stored as JSON string
    profile_image = db.Column(db.String(255))

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(20), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(50), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    parent_phone = db.Column(db.String(20), nullable=False)
    face_encoding = db.Column(db.Text) # Stored as JSON string
    profile_image = db.Column(db.String(255))

class Subject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject_code = db.Column(db.String(20), unique=True, nullable=False)
    subject_name = db.Column(db.String(100), nullable=False)
    faculty_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    schedules = db.relationship('SubjectSchedule', backref='subject', lazy=True)

class SubjectSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    day = db.Column(db.String(20), nullable=False) # e.g., 'Monday'
    start_time = db.Column(db.Time, nullable=False)
    end_time = db.Column(db.Time, nullable=False)
    classroom_name = db.Column(db.String(50)) # e.g., 'CSBS6'

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=lambda: get_ist_time().date())
    time = db.Column(db.Time, nullable=False, default=lambda: get_ist_time().time())
    status = db.Column(db.Enum('Present', 'Absent'), default='Present')

class TeacherAttendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False, default=lambda: get_ist_time().date())
    time = db.Column(db.Time, nullable=False, default=lambda: get_ist_time().time())
    status = db.Column(db.Enum('Present', 'Absent'), default='Present')

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Communication Utils ---
def send_sms(to_number, message):
    try:
        if TWILIO_ACCOUNT_SID != 'your_sid':
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=to_number)
            print(f"REAL SMS SENT to {to_number}: {message}")
        else:
            print(f"[DEVELOPMENT MODE] SMS sent to {to_number}: {message}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

def send_whatsapp(to_number, message, media_url=None):
    """Integrates WhatsApp Business API via Twilio"""
    try:
        if TWILIO_ACCOUNT_SID != 'your_sid':
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            # Format numbers for WhatsApp
            from_wa = f"whatsapp:{TWILIO_PHONE_NUMBER}"
            to_wa = f"whatsapp:{to_number}"
            
            if media_url:
                client.messages.create(body=message, from_=from_wa, to=to_wa, media_url=[media_url])
            else:
                client.messages.create(body=message, from_=from_wa, to=to_wa)
            print(f"REAL WHATSAPP SENT to {to_number}: {message}")
        else:
            print(f"[DEVELOPMENT MODE] WhatsApp (Bot) to {to_number}: {message}")
    except Exception as e:
        print(f"Failed to send WhatsApp message: {e}")

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and bcrypt.check_password_hash(user.password_hash, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# --- Attendance Logic ---
# --- SHARED ASYNCHRONOUS CAMERA ENGINE ---
# --- SHARED ASYNCHRONOUS CAMERA ENGINE (Advanced Hardware Wakeup) ---
# --- SHARED ASYNCHRONOUS CAMERA ENGINE (Ultra-Robust Hardware Wakeup) ---
class CameraStream:
    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.last_frame = None
        self.is_running = False
        self.thread = None
        self.failed_indices = set()

    def _capture_loop(self):
        """Background thread that constantly pumps frames and monitors for black screens."""
        black_frame_count = 0
        restart_attempts = 0
        MAX_RESTART_ATTEMPTS = 3
        
        while self.is_running and restart_attempts < MAX_RESTART_ATTEMPTS:
            try:
                if self.cap and self.cap.isOpened():
                    success, frame = self.cap.read()
                    if success and frame is not None:
                        brightness = np.mean(frame)
                        if brightness < 1.0: # Absolute black
                            black_frame_count += 1
                            if black_frame_count > 60: # 2 seconds of blackness
                                app.logger.warning("Camera went black. Resetting hardware link...")
                                self.is_running = False # Trigger exit
                                break
                        else:
                            black_frame_count = 0
                        
                        with self.lock:
                            self.last_frame = frame.copy()
                    else:
                        app.logger.warning("Frame capture lost.")
                        self.is_running = False
                        break
                else:
                    self.is_running = False
                    break
                time.sleep(0.01)
                
            except Exception as e:
                app.logger.error(f"CAMERA THREAD ERROR: {e}", exc_info=True)
                time.sleep(1)
                restart_attempts += 1
                
                if self.is_running and restart_attempts < MAX_RESTART_ATTEMPTS:
                    app.logger.info(f"Attempting camera restart ({restart_attempts}/{MAX_RESTART_ATTEMPTS})...")
                    time.sleep(2)
                    self.is_running = True  # Allow retry
                else:
                    self.is_running = False
                    break
    
    def stop(self):
        """Safely stop the camera stream."""
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            if self.thread:
                self.thread.join(timeout=5)
                self.thread = None

    def start(self):
        with self.lock:
            if self.is_running: return True
            
            # Try finding a camera that actually SENDS data
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
            for index in [0, 1]:
                if index in self.failed_indices and index != 0: continue
                
                for backend in backends:
                    print(f"DEBUG: Attempting Wakeup - Index {index}, Backend {backend}")
                    if backend is not None:
                        self.cap = cv2.VideoCapture(index, backend)
                    else:
                        self.cap = cv2.VideoCapture(index)
                    
                    if self.cap.isOpened():
                        # Set standard resolution
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        
                        # WARMUP & POKE: Read up to 60 frames to get a picture
                        for i in range(60):
                            success, frame = self.cap.read()
                            if success and frame is not None:
                                brightness = np.mean(frame)
                                if brightness > 1.0: # Valid picture found
                                    self.is_running = True
                                    self.thread = threading.Thread(target=self._capture_loop, daemon=True)
                                    self.thread.start()
                                    print(f"HARDWARE LINK ESTABLISHED: Index {index}, Backend {backend}")
                                    return True
                                
                                # If after 15 frames it's still black, try to 'jolt' the sensor
                                if i == 15:
                                    print("DEBUG: Sensor is black. Hardware properties toggle skipped for stability.")
                                    # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
                                    # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Auto
                                    # self.cap.set(cv2.CAP_PROP_EXPOSURE, -5) # Force some exposure
                            
                        # If still black, release and try next backend
                        self.cap.release()
            
            print("CRITICAL: All camera backends failed to produce image data.")
            return False

    def get_frame(self):
        if not self.is_running: 
            # Non-blocking start attempt
            threading.Thread(target=self.start, daemon=True).start()
            return None
            
        with self.lock:
            return self.last_frame if self.last_frame is not None else None


global_stream = CameraStream()

def generate_attendance_frames(subject_id):
    if not global_stream.start():
        print("CRITICAL: Camera Engine Failed.")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "CAMERA NOT FOUND", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    try:
        with app.app_context():
            # Load potential matches efficiently
            students = Student.query.all()
            faculties = User.query.filter_by(role='faculty').all()
            known_encodings, known_ids, id_to_name = [], [], {}
            
            for s in students:
                if s.face_encoding:
                    try:
                        enc = json.loads(s.face_encoding)
                        known_encodings.append(enc)
                        sid = f"STUDENT_{s.id}"
                        known_ids.append(sid)
                        id_to_name[sid] = s.name
                    except json.JSONDecodeError as e:
                        app.logger.error(f"Malformed face encoding for student {s.id}: {e}")
                        continue
            
            for f in faculties:
                if f.face_encoding:
                    try:
                        enc = json.loads(f.face_encoding)
                        known_encodings.append(enc)
                        fid = f"FACULTY_{f.id}"
                        known_ids.append(fid)
                        id_to_name[fid] = f.full_name
                    except json.JSONDecodeError as e:
                        app.logger.error(f"Malformed face encoding for faculty {f.id}: {e}")
                        continue

            frame_count = 0
            face_locations, face_names = [], []
            student_marked_ids, faculty_marked_ids = set(), set()
            match_votes = {} 

            while True:
                frame = global_stream.get_frame()
                if frame is None:
                    # Let the camera try to wake up
                    time.sleep(0.5)
                    frame = global_stream.get_frame()
                    if frame is None:
                        # Yield a "reconnecting" frame instead of breaking the stream
                        reconn = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(reconn, "WAKING UP CAMERA...", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        _, buffer = cv2.imencode('.jpg', reconn)
                        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                        continue
                
                frame_count += 1
                
                # STABLE ENGINE: Process every 5th frame for rock-solid tracking
                if frame_count % 5 == 0:
                    try:
                        # Safety check for libraries
                        if not FACE_RECOGNITION_AVAILABLE:
                            face_names = ["ERROR: face_recognition missing"]
                        else:
                            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                            rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                            face_names = []
                            
                            rgb_full = np.ascontiguousarray(frame[:, :, ::-1])
                            
                            for (top, right, bottom, left) in face_locations:
                                orig_top, orig_right, orig_bottom, orig_left = top*4, right*4, bottom*4, left*4
                                face_rect = dlib.rectangle(orig_left, orig_top, orig_right, orig_bottom)
                                
                                if verify_liveness(frame, face_rect):
                                    recon_id = recognize_face_at_location(rgb_full, (orig_top, orig_right, orig_bottom, orig_left), known_encodings, known_ids)
                                    
                                    if recon_id:
                                        # Temporal Smoothing: 3-vote rule
                                        match_votes[recon_id] = match_votes.get(recon_id, 0) + 1
                                        actual_name = id_to_name.get(recon_id, "User")

                                        if match_votes[recon_id] >= 3:
                                            status_tag = f"{actual_name} [VERIFIED]"
                                            if recon_id.startswith("STUDENT_"):
                                                student_db_id = int(recon_id.split("_")[1])
                                                ist_now = get_ist_time()
                                                if student_db_id not in student_marked_ids:
                                                    try:
                                                        new_attendance = Attendance(
                                                            student_id=student_db_id,
                                                            subject_id=subject_id,
                                                            date=ist_now.date(),
                                                            time=ist_now.time(),
                                                            status='Present'
                                                        )
                                                        db.session.add(new_attendance)
                                                        db.session.flush()
                                                        db.session.commit()
                                                        student_marked_ids.add(student_db_id)
                                                        app.logger.info(f"Attendance recorded for student {student_db_id}")
                                                    except IntegrityError:
                                                        db.session.rollback()
                                                        app.logger.debug(f"Attendance already recorded for student {student_db_id}")
                                                    except Exception as e:
                                                        db.session.rollback()
                                                        app.logger.error(f"Error recording attendance: {e}", exc_info=True)
                                            elif recon_id.startswith("FACULTY_"):
                                                faculty_db_id = int(recon_id.split("_")[1])
                                                try:
                                                    faculty_user = db.session.get(User, faculty_db_id)
                                                    if faculty_user:
                                                        live_faculty_sessions[subject_id] = {'name': faculty_user.full_name, 'time': get_ist_time().strftime('%I:%M:%S %p'), 'timestamp': time.time()}
                                                    faculty_marked_ids.add(faculty_db_id)
                                                except Exception as e:
                                                    app.logger.error(f"Error processing faculty attendance: {e}", exc_info=True)
                                            face_names.append(status_tag)
                                        else:
                                            face_names.append(f"{actual_name} (Verifying...)")
                                    else:
                                        face_names.append("SEARCHING...")
                                        match_votes = {}
                                else:
                                    face_names.append("CENTER FACE...")
                                    match_votes = {}
                    except Exception as loop_e:
                        print(f"Loop Engine Error: {loop_e}")

                # DRAWING: Sharper, vibrantly color-coded overlays
                if len(face_locations) == len(face_names):
                    for (location, name) in zip(face_locations, face_names):
                        t, r, b, l = [c * 4 for c in location]
                        color = (0, 255, 0) if "[VERIFIED]" in name else (0, 0, 255) if "SEARCHING" in name else (0, 165, 255)
                        cv2.rectangle(frame, (l, t), (r, b), color, 2)
                        cv2.putText(frame, name, (l, t-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except Exception as e:
        print(f"Generator Error: {e}")
            
    finally:
        # Note: We don't stop the global stream here to keep it ready for other routes
        pass

@app.route('/video_feed/<int:subject_id>')
@login_required
def video_feed(subject_id):
    return Response(generate_attendance_frames(subject_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_attendance/<int:subject_id>')
@login_required
def start_attendance(subject_id):
    subject = db.session.get(Subject, subject_id)
    if not subject:
        flash('Subject not found', 'danger')
        return redirect(url_for('dashboard'))
    is_room_acct = current_user.username.startswith('CSBS')
    return render_template('attendance_session.html', subject=subject, is_room_acct=is_room_acct)

# --- FACULTY STREAM ENGINE ---
def generate_teacher_attendance_frames():
    if not global_stream.start():
        print("CRITICAL: Teacher Streaming Engine Failed.")
        return

    try:
        with app.app_context():
            teachers = User.query.filter_by(role='faculty').all()
            known_encodings, known_ids, match_votes = [], [], {}
            for u in teachers:
                if u.face_encoding:
                    try:
                        enc = json.loads(u.face_encoding)
                        known_encodings.append(enc)
                        known_ids.append(u.id)
                    except: continue

            frame_count = 0
            face_locations, face_names = [], []
            session_marked = False

            while True:
                frame = global_stream.get_frame()
                if frame is None: break
                frame_count += 1
                
                # STABLE RELIABILITY: Every 6th frame for lower CPU
                if frame_count % 6 == 0:
                    try:
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
                        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                        face_names = []
                        
                        for (top, right, bottom, left) in face_locations:
                            orig_top, orig_right, orig_bottom, orig_left = top*4, right*4, bottom*4, left*4
                            face_rect = dlib.rectangle(orig_left, orig_top, orig_right, orig_bottom)
                            
                            if verify_liveness(frame, face_rect):
                                rgb_full = np.ascontiguousarray(frame[:, :, ::-1])
                                teacher_db_id = recognize_face_at_location(rgb_full, (orig_top, orig_right, orig_bottom, orig_left), known_encodings, known_ids)
                                if teacher_db_id:
                                    match_votes[teacher_db_id] = match_votes.get(teacher_db_id, 0) + 1
                                    if match_votes[teacher_db_id] >= 3:
                                        face_names.append("TEACHER [VERIFIED]")
                                        if not session_marked:
                                            with app.app_context():
                                                ist_now = get_ist_time()
                                                if not TeacherAttendance.query.filter_by(user_id=teacher_db_id, date=ist_now.date()).first():
                                                    db.session.add(TeacherAttendance(user_id=teacher_db_id, date=ist_now.date(), time=ist_now.time()))
                                                    db.session.commit()
                                            session_marked = True
                                    else:
                                        face_names.append("Matching...")
                                else:
                                    face_names.append("UNKNOWN")
                                    match_votes = {}
                            else:
                                face_names.append("VERIFYING...")
                                match_votes = {}
                    except Exception as loop_e:
                        print(f"Teacher Engine error: {loop_e}")

                for (location, name) in zip(face_locations, face_names):
                    t, r, b, l = [c * 4 for c in location]
                    color = (0, 255, 0) if "[VERIFIED]" in name else (0, 0, 255) if "UNKNOWN" in name else (0, 165, 255)
                    cv2.rectangle(frame, (l, t), (r, b), color, 2)
                    cv2.putText(frame, name, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        pass

@app.route('/teacher_video_feed')
@login_required
def teacher_video_feed():
    return Response(generate_teacher_attendance_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/faculty_attendance')
@login_required
def faculty_attendance():
    return render_template('faculty_attendance_session.html')

@app.route('/api/check_teacher_attendance')
@login_required
def check_teacher_attendance():
    db.session.remove() 
    today = get_ist_time().date()
    
    # If the user is a teacher, they only care about THEIR OWN status (marked in last 15s)
    if current_user.role == 'faculty':
        ist_now = get_ist_time()
        time_threshold = (ist_now - timedelta(seconds=15)).time()
        attendance = db.session.query(TeacherAttendance).filter(
            TeacherAttendance.user_id == current_user.id,
            TeacherAttendance.date == today,
            TeacherAttendance.time >= time_threshold
        ).first()
        if attendance:
            return jsonify({'status': 'present', 'name': current_user.full_name, 'message': f'{current_user.full_name} is Verified'})
    
    # For Admins: Show if ANY faculty was marked recently (last 15 seconds)
    if current_user.role == 'admin':
        ist_now = get_ist_time()
        time_threshold = (ist_now - timedelta(seconds=15)).time()
        
        last_any = db.session.query(TeacherAttendance, User).join(User).filter(
            TeacherAttendance.date == today,
            TeacherAttendance.time >= time_threshold
        ).order_by(TeacherAttendance.id.desc()).first()
        if last_any:
            return jsonify({'status': 'present', 'name': last_any.User.full_name, 'message': f'{last_any.User.full_name} is Verified'})
            
    return jsonify({'status': 'absent'})

@app.route('/api/check_student_presence/<int:subject_id>')
@login_required
def check_student_presence(subject_id):
    today = get_ist_time().date()
    
    # 1. Faculty/Teacher live presence (shared with everyone)
    teacher_data = None
    live_info = live_faculty_sessions.get(subject_id)
    if live_info:
        if time.time() - live_info.get('timestamp', 0) < 60:
            teacher_data = live_info

    # 2. Check if this is a regular student OR a monitoring account
    is_room_acct = current_user.username.startswith('CSBS')
    is_monitor = current_user.role in ['admin', 'faculty'] or is_room_acct
    
    # Logic for individual students: ONLY return their own status
    if current_user.role == 'student' and not is_room_acct:
        student = Student.query.filter_by(student_id=current_user.username).first()
        if student:
            attendance = Attendance.query.filter_by(student_id=student.id, subject_id=subject_id, date=today).first()
            if attendance:
                # PRECISION: If Suhas is logged in, he only sees his name here
                return jsonify({'status': 'present', 'name': student.name, 'faculty': teacher_data})
            else:
                return jsonify({'status': 'absent', 'faculty': teacher_data})

    # Logic for Monitoring Accounts (Admin, Faculty, Room Account): Show recent activity
    if is_monitor:
        # ABSOLUTE SYNC: Get the very last mark for today
        last_mark = db.session.query(Attendance, Student).join(Student).filter(
            Attendance.subject_id == subject_id, 
            Attendance.date == today
        ).order_by(Attendance.id.desc()).first()
        
        if last_mark:
            return jsonify({
                'status': 'recent', 
                'name': last_mark.Student.name,
                'faculty': teacher_data
            })
        
    return jsonify({'status': 'none', 'faculty': teacher_data})

@app.route('/register_student', methods=['GET', 'POST'])
@login_required
def register_student():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        try:
            student_id = request.form['student_id']
            name = request.form['name']
            dept = request.form['department']
            year = request.form['year']
            phone = request.form['phone']
            
            # Handle face image
            if 'face_image' not in request.files:
                flash('No image file selected.', 'danger')
                return redirect(request.url)
                
            file = request.files['face_image']
            if file.filename == '':
                flash('No selected file.', 'danger')
                return redirect(request.url)

            if file:
                # Ensure upload directory exists
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'students')
                os.makedirs(upload_dir, exist_ok=True)
                filename = f"{student_id}.jpg"
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                
                # Extract encoding
                encoding = get_face_encodings(filepath)
                if encoding:
                    new_student = Student(
                        student_id=student_id,
                        name=name,
                        department=dept,
                        year=year,
                        parent_phone=phone,
                        face_encoding=json.dumps(encoding),
                        profile_image=f"students/{filename}"
                    )
                    db.session.add(new_student)
                    db.session.commit()
                    flash(f'Student {name} registered successfully!', 'success')
                    return redirect(url_for('manage_students'))
                else:
                    # If encoding fails, delete the uploaded file to keep system clean
                    os.remove(filepath)
                    flash('Error: No face detected in the image. Please use a clear front-facing photo.', 'danger')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'danger')
            print(f"Registration Error: {e}")
        
    return render_template('register_student.html')

@app.route('/manage_faculty')
@login_required
def manage_faculty():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    faculties = User.query.filter_by(role='faculty').all()
    return render_template('manage_faculty.html', faculties=faculties)

@app.route('/register_faculty', methods=['GET', 'POST'])
@login_required
def register_faculty():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        try:
            username = request.form['username']
            full_name = request.form['full_name']
            password = request.form['password']
            
            if 'face_image' not in request.files:
                flash('No image selected', 'danger')
                return redirect(request.url)
            
            file = request.files['face_image']
            if file:
                upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'faculty')
                os.makedirs(upload_dir, exist_ok=True)
                filename = f"{username}.jpg"
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                
                encoding = get_face_encodings(filepath)
                if encoding:
                    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
                    new_faculty = User(
                        username=username,
                        full_name=full_name,
                        password_hash=hashed_pw,
                        role='faculty',
                        face_encoding=json.dumps(encoding),
                        profile_image=f"faculty/{filename}"
                    )
                    db.session.add(new_faculty)
                    db.session.commit()
                    flash(f'Faculty {full_name} registered successfully!', 'success')
                    return redirect(url_for('manage_faculty'))
                else:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    flash('No face detected', 'danger')
        except Exception as e:
            flash(f'Error: {str(e)}', 'danger')
            
    return render_template('register_faculty.html')
@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    # Only admins can access registration
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        try:
            username = request.form['username']
            full_name = request.form['full_name']
            password = request.form['password']
            role = request.form['role']
            
            # Check if user already exists
            if User.query.filter_by(username=username).first():
                flash('Username/ID already registered in the system.', 'danger')
                return redirect(url_for('register'))

            if 'face_image' not in request.files:
                flash('Biometric verification (Face Image) is mandatory.', 'danger')
                return redirect(request.url)
            
            file = request.files['face_image']
            if file and file.filename != '':
                # Save path configuration
                subfolder = 'faculty' if role == 'faculty' else 'students'
                upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], subfolder)
                os.makedirs(upload_dir, exist_ok=True)
                
                filename = f"{username}.jpg"
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                
                # Biometric Encoding extraction
                encoding = get_face_encodings(filepath)
                if encoding:
                    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
                    
                    # 1. Create the base User account
                    new_user = User(
                        username=username,
                        full_name=full_name,
                        password_hash=hashed_pw,
                        role=role,
                        face_encoding=json.dumps(encoding),
                        profile_image=f"{subfolder}/{filename}"
                    )
                    db.session.add(new_user)
                    
                    # 2. If it's a student, we MUST also create the Student metadata entry
                    if role == 'student':
                        dept = request.form.get('department', 'GEN')
                        year = request.form.get('year', 1)
                        phone = request.form.get('phone', 'N/A')
                        
                        student_entry = Student(
                            student_id=username, # USN matches username
                            name=full_name,
                            department=dept,
                            year=year,
                            parent_phone=phone,
                            face_encoding=json.dumps(encoding),
                            profile_image=filename
                        )
                        db.session.add(student_entry)
                    
                    db.session.commit()
                    flash(f'Neuro-profile generated! You can now authorize access as {full_name}.', 'success')
                    return redirect(url_for('login'))
                else:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    flash('Neuro-biometric scan failed. Ensure your face is clearly visible without obstructions.', 'danger')
        except Exception as e:
            db.session.rollback()
            flash(f'Neuro-biometric Error: {str(e)}', 'danger')
            print(f"Registration Error: {e}")
            
    return render_template('register.html')

@app.route('/attendance_report/<int:student_id>')
@login_required
def attendance_report(student_id):
    student = Student.query.get_or_404(student_id)
    # Calculate subject-wise percentage
    subjects = Subject.query.all()
    report = []
    
    for sub in subjects:
        total_classes = Attendance.query.filter_by(subject_id=sub.id).count()
        attended = Attendance.query.filter_by(student_id=student.id, subject_id=sub.id, status='Present').count()
        
        percentage = (attended / total_classes * 100) if total_classes > 0 else 0
        report.append({
            'subject': sub.subject_name,
            'percentage': round(percentage, 2),
            'status': 'Low' if percentage < 75 else 'Good'
        })
        
        # Trigger WhatsApp/SMS if low
        if percentage < 75 and total_classes > 5:
            msg = f"🔍 *SmartAttend.AI Alert*\n\nStudent: *{student.name}*\nSubject: {sub.subject_name}\nStatus: *{round(percentage, 2)}%* (Below Threshold)\n\nPlease ensure regular attendance to avoid academic penalty."
            # Prioritize WhatsApp for detailed alerts
            send_whatsapp(student.parent_phone, msg)
            
    return jsonify(report)

@app.route('/send_pdf_report/<int:student_id>')
@login_required
def send_pdf_report(student_id):
    student = Student.query.get_or_404(student_id)
    subjects = Subject.query.all()
    report = []
    
    for sub in subjects:
        total_classes = Attendance.query.filter_by(subject_id=sub.id).count()
        attended = Attendance.query.filter_by(student_id=student.id, subject_id=sub.id, status='Present').count()
        perc = (attended / total_classes * 100) if total_classes > 0 else 0
        report.append({
            'subject': sub.subject_name,
            'percentage': round(perc, 2),
            'status': 'Low' if perc < 75 else 'Good'
        })
    
    # Generate PDF
    reports_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    filename = f"Report_{student.student_id}_{get_ist_time().strftime('%Y%m%d')}.pdf"
    filepath = os.path.join(reports_dir, filename)
    
    generate_student_report_pdf(student.name, student.student_id, report, filepath)
    
    # Send WhatsApp notification that report is ready
    wa_msg = f"📋 *Academic Report Ready*\n\nDetailed attendance report for *{student.name}* has been generated.\n\n_Note: In professional environments, this PDF is hosted on a secure portal for download._"
    send_whatsapp(student.parent_phone, wa_msg)
    
    flash(f'PDF Report for {student.name} generated and WhatsApp alert sent!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/manage_students')
@login_required
def manage_students():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    students = Student.query.all()
    return render_template('manage_students.html', students=students)

@app.route('/manage_subjects', methods=['GET', 'POST'])
@login_required
def manage_subjects():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        code = request.form['code']
        name = request.form['name']
        faculty_id = request.form['faculty_id']
        new_subject = Subject(subject_code=code, subject_name=name, faculty_id=faculty_id)
        db.session.add(new_subject)
        db.session.commit()
        flash('Subject added successfully!', 'success')
        
    subjects = Subject.query.all()
    faculties = User.query.filter_by(role='faculty').all()
    return render_template('manage_subjects.html', subjects=subjects, faculties=faculties)

@app.route('/add_schedule', methods=['POST'])
@login_required
def add_schedule():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    subject_id = request.form['subject_id']
    day = request.form['day']
    room = request.form['classroom_name']
    try:
        start_time = datetime.strptime(request.form['start_time'], '%H:%M').time()
        end_time = datetime.strptime(request.form['end_time'], '%H:%M').time()
        
        new_schedule = SubjectSchedule(
            subject_id=subject_id,
            day=day,
            start_time=start_time,
            end_time=end_time,
            classroom_name=room
        )
        db.session.add(new_schedule)
        db.session.commit()
        flash('New class added to time table!', 'success')
    except Exception as e:
        flash(f'Error adding schedule: {str(e)}', 'danger')
        
    return redirect(url_for('manage_subjects'))

@app.route('/delete_schedule/<int:id>')
@login_required
def delete_schedule(id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    schedule = SubjectSchedule.query.get_or_404(id)
    db.session.delete(schedule)
    db.session.commit()
    flash('Session removed from timetable', 'success')
    return redirect(url_for('manage_subjects'))

@app.route('/delete_subject/<int:id>')
@login_required
def delete_subject(id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    subject = Subject.query.get_or_404(id)
    # Cleanup: Delete all schedules and attendance for this subject
    SubjectSchedule.query.filter_by(subject_id=id).delete()
    Attendance.query.filter_by(subject_id=id).delete()
    
    db.session.delete(subject)
    db.session.commit()
    flash(f'Module {subject.subject_name} and all associated logs were deleted.', 'success')
    return redirect(url_for('manage_subjects'))

@app.route('/dashboard')
@login_required
def dashboard():
    today_day = get_ist_time().strftime('%A')
    
    if current_user.role == 'admin':
        num_students = Student.query.count()
        num_subjects = Subject.query.count()
        recent_attendance = db.session.query(Attendance, Student, Subject).join(Student).join(Subject).order_by(Attendance.id.desc()).limit(10).all()
        recent_teacher_attendance = db.session.query(TeacherAttendance, User).join(User).order_by(TeacherAttendance.id.desc()).limit(10).all()
        
        # Admin sees all classes for today
        today_schedules = db.session.query(SubjectSchedule, Subject).join(Subject).filter(SubjectSchedule.day == today_day).order_by(SubjectSchedule.start_time).all()
        
        # --- NEW: Attendance Analytics for Charts ---
        today = get_ist_time().date()
        total_present_today = db.session.query(Attendance.student_id).filter(Attendance.date == today).distinct().count()
        total_absent_today = max(0, num_students - total_present_today)
        
        # Subject-wise distribution
        subject_stats = db.session.query(Subject.subject_name, db.func.count(Attendance.id)).join(Attendance).filter(Attendance.date == today).group_by(Subject.id).all()
        subject_labels = [s[0] for s in subject_stats]
        subject_counts = [s[1] for s in subject_stats]

        return render_template('admin_dashboard.html', 
                               num_students=num_students, 
                               num_subjects=num_subjects, 
                               recent_attendance=recent_attendance, 
                               recent_teacher_attendance=recent_teacher_attendance,
                               today_schedules=today_schedules,
                               present_count=total_present_today,
                               absent_count=total_absent_today,
                               subject_labels=subject_labels,
                               subject_counts=subject_counts)

    elif current_user.role == 'faculty':
        subjects = Subject.query.filter_by(faculty_id=current_user.id).all()
        # Faculty sees only their classes today
        today_schedules = db.session.query(SubjectSchedule, Subject).join(Subject).filter(
            Subject.faculty_id == current_user.id,
            SubjectSchedule.day == today_day
        ).order_by(SubjectSchedule.start_time).all()
        
        return render_template('faculty_dashboard.html', subjects=subjects, today_schedules=today_schedules)
    
    # Student dashboard
    student = Student.query.filter_by(student_id=current_user.username).first()
    
    # Check if this is a classroom-specific login (like CSBS6)
    is_room_acct = current_user.username.startswith('CSBS')
    
    if student or is_room_acct:
        # Correctly use student.id for foreign key lookup
        attendances = Attendance.query.filter_by(student_id=student.id).all() if student else []
        
        # Room accounts only see their own room's schedule
        if is_room_acct:
            today_schedules = db.session.query(SubjectSchedule, Subject).join(Subject).filter(
                SubjectSchedule.day == today_day,
                SubjectSchedule.classroom_name == current_user.username
            ).order_by(SubjectSchedule.start_time).all()
        else:
            today_schedules = db.session.query(SubjectSchedule, Subject).join(Subject).filter(SubjectSchedule.day == today_day).order_by(SubjectSchedule.start_time).all()
        
        # --- FIX: Identify Current/Next Schedule for UI Focus ---
        ist_now = get_ist_time().time()
        current_schedule = None
        
        # 1. Look for currently active session
        for entry in today_schedules:
            if entry.SubjectSchedule.start_time <= ist_now <= entry.SubjectSchedule.end_time:
                current_schedule = entry
                break
        
        # 2. If no active session, look for the next session
        if not current_schedule:
            for entry in today_schedules:
                if entry.SubjectSchedule.start_time > ist_now:
                    current_schedule = entry
                    break
        
        # 3. Fallback to first session if none found (or stay None)
        if not current_schedule and today_schedules:
            current_schedule = today_schedules[0]

        return render_template('student_dashboard.html', 
                               student=student or current_user, 
                               attendances=attendances, 
                               today_schedules=today_schedules, 
                               current_schedule=current_schedule,
                               is_room_acct=is_room_acct)
    
    return render_template('student_dashboard.html')

@app.route('/export_student_attendance')
@login_required
def export_student_attendance():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    # Pre-calculate percentages to avoid N+1 query problem
    student_subject_stats = {}
    
    # Query all attendance
    all_attendance = db.session.query(Attendance, Student, Subject).join(Student).join(Subject).all()
    
    # Calculate stats
    subjects = Subject.query.all()
    for sub in subjects:
        total_classes = Attendance.query.filter_by(subject_id=sub.id).count()
        if total_classes == 0: continue
        
        students_attended = db.session.query(Attendance.student_id, db.func.count(Attendance.id)).filter(
            Attendance.subject_id == sub.id, Attendance.status == 'Present'
        ).group_by(Attendance.student_id).all()
        
        for stu_id, count in students_attended:
            perc = (count / total_classes) * 100
            student_subject_stats[(stu_id, sub.id)] = f"{round(perc, 2)}%"

    report_data = []
    for att, stu, sub in all_attendance:
        percentage = student_subject_stats.get((stu.id, sub.id), "0.0%")
        
        report_data.append({
            'Student Name': stu.name,
            'USN': stu.student_id,
            'Subject': sub.subject_name,
            'Subject Code': sub.subject_code,
            'Date': att.date.strftime('%Y-%m-%d'),
            'Time': att.time.strftime('%H:%M:%S'),
            'Attended So Far (%)': percentage
        })
    
    if not report_data:
        report_data = [{'Status': 'No attendance records found'}]

    df = pd.DataFrame(report_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Detailed Attendance')
    
    output.seek(0)
    return Response(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-disposition": "attachment; filename=Student_Attendance_Report.xlsx"}
    )

@app.route('/delete_student/<int:id>')
@login_required
def delete_student(id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    student = Student.query.get_or_404(id)
    if student.profile_image:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], student.profile_image))
        except: pass
    Attendance.query.filter_by(student_id=student.id).delete()
    db.session.delete(student)
    db.session.commit()
    flash('Student record deleted successfully', 'success')
    return redirect(url_for('manage_students'))

@app.route('/delete_faculty/<int:id>')
@login_required
def delete_faculty(id):
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    faculty = User.query.get_or_404(id)
    if faculty.profile_image:
        try:
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], faculty.profile_image.split('/')[-1]))
        except: pass
    TeacherAttendance.query.filter_by(user_id=faculty.id).delete()
    db.session.delete(faculty)
    db.session.commit()
    flash('Faculty record deleted successfully', 'success')
    return redirect(url_for('manage_faculty'))

@app.route('/export_teacher_attendance')
@login_required
def export_teacher_attendance():
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))
    
    data = db.session.query(TeacherAttendance, User).join(User).all()
    
    report_data = []
    for att, user in data:
        report_data.append({
            'Teacher Name': user.full_name,
            'Username': user.username,
            'Date': att.date.strftime('%Y-%m-%d'),
            'Time': att.time.strftime('%H:%M:%S'),
            'Status': att.status
        })
    
    df = pd.DataFrame(report_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Teacher Attendance')
    
    output.seek(0)
    return Response(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-disposition": "attachment; filename=Teacher_Attendance_Report.xlsx"}
    )

if __name__ == '__main__':
    # Ensure all upload subdirectories exist
    for sub in ['students', 'faculty']:
        path = os.path.join(app.config['UPLOAD_FOLDER'], sub)
        os.makedirs(path, exist_ok=True)
    with app.app_context():
        db.create_all()
        # Create default classroom user if not exists
        if not User.query.filter_by(username='CSBS6').first():
            hashed_pw = bcrypt.generate_password_hash('password123').decode('utf-8')
            new_user = User(username='CSBS6', password_hash=hashed_pw, role='student', full_name='Classroom CSBS6')
            db.session.add(new_user)
            db.session.commit()
            print("Default classroom user 'CSBS6' created (Password: password123)")
    app.run(debug=True)
