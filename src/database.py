"""
Database operations for face attendance system
"""
import sqlite3
import pickle
import numpy as np
from datetime import datetime, timedelta
import os
import pytz

class AttendanceDatabase:
    """SQLite database for user and attendance management"""
    
    def __init__(self, db_path):
        """Initialize database connection"""
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _get_current_ist_time(self):
        """Get current time in IST"""
        ist = pytz.timezone('Asia/Kolkata')
        return datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
    
    def _create_tables(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table - changed to NOT NULL for timestamps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                employee_id TEXT UNIQUE,
                department TEXT,
                embedding BLOB NOT NULL,
                num_images INTEGER DEFAULT 0,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        ''')
        
        # Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                punch_type TEXT NOT NULL CHECK(punch_type IN ('IN', 'OUT')),
                timestamp TIMESTAMP NOT NULL,
                confidence_score REAL,
                status TEXT DEFAULT 'PRESENT',
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_user(self, name, employee_id, department, embedding, num_images):
        """Add new user to database with IST timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            embedding_blob = pickle.dumps(embedding)
            
            # Get current IST time
            current_time = self._get_current_ist_time()
            
            cursor.execute('''
                INSERT INTO users (name, employee_id, department, embedding, num_images, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (name, employee_id, department, embedding_blob, num_images, current_time, current_time))
            
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
            
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def get_all_users(self):
        """Get all registered users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, name, employee_id, department, embedding, num_images, created_at
            FROM users
            ORDER BY created_at DESC
        ''')
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'user_id': row[0],
                'name': row[1],
                'employee_id': row[2],
                'department': row[3],
                'embedding': pickle.loads(row[4]),
                'num_images': row[5],
                'created_at': row[6]
            })
        
        conn.close()
        return users
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT user_id, name, employee_id, department, embedding
            FROM users WHERE user_id = ?
        ''', (user_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'user_id': row[0],
                'name': row[1],
                'employee_id': row[2],
                'department': row[3],
                'embedding': pickle.loads(row[4])
            }
        return None
    
    def delete_user(self, user_id):
        """Delete user from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM users WHERE user_id = ?', (user_id,))
        cursor.execute('DELETE FROM attendance WHERE user_id = ?', (user_id,))
        
        conn.commit()
        conn.close()
    
    def find_matching_user(self, test_embedding, threshold=0.7):
        """Find matching user by comparing embeddings"""
        users = self.get_all_users()
        
        if not users:
            return None
        
        best_match = None
        best_similarity = -1
        
        for user in users:
            # Cosine similarity
            similarity = np.dot(test_embedding, user['embedding']) / (
                np.linalg.norm(test_embedding) * np.linalg.norm(user['embedding'])
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user
                best_match['similarity'] = similarity
        
        if best_similarity >= threshold:
            return best_match
        return None
    
    def mark_attendance(self, user_id, punch_type, confidence_score):
        """Mark attendance with IST timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current IST time
        current_time = self._get_current_ist_time()
        
        cursor.execute('''
            INSERT INTO attendance (user_id, punch_type, confidence_score, timestamp)
            VALUES (?, ?, ?, ?)
        ''', (user_id, punch_type, float(confidence_score), current_time))
        
        attendance_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return attendance_id
    
    def get_last_attendance(self, user_id):
        """Get last attendance record for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT attendance_id, punch_type, timestamp, confidence_score
            FROM attendance
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'attendance_id': result[0],
                'punch_type': result[1],
                'timestamp': result[2],
                'confidence_score': result[3]
            }
        return None
    
    def get_attendance_records(self, user_id=None, date=None, limit=50):
        """Get attendance records with filters"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT 
                a.attendance_id,
                u.name,
                u.employee_id,
                u.department,
                a.punch_type,
                a.timestamp,
                a.confidence_score
            FROM attendance a
            JOIN users u ON a.user_id = u.user_id
            WHERE 1=1
        '''
        
        params = []
        
        if user_id:
            query += ' AND a.user_id = ?'
            params.append(user_id)
        
        if date:
            # Convert date to string for comparison
            if isinstance(date, datetime):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            query += ' AND DATE(a.timestamp) = ?'
            params.append(date_str)
        
        query += ' ORDER BY a.timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        
        records = []
        for row in cursor.fetchall():
            records.append({
                'attendance_id': row[0],
                'name': row[1],
                'employee_id': row[2],
                'department': row[3],
                'punch_type': row[4],
                'timestamp': row[5],
                'confidence_score': row[6]
            })
        
        conn.close()
        return records
    
    def get_daily_report(self, date=None):
        """Generate daily attendance report"""
        if date is None:
            ist = pytz.timezone('Asia/Kolkata')
            date = datetime.now(ist).date()
        
        # Convert date to string
        if isinstance(date, datetime):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                u.user_id,
                u.name,
                u.employee_id,
                u.department,
                a.punch_type,
                a.timestamp
            FROM attendance a
            JOIN users u ON a.user_id = u.user_id
            WHERE DATE(a.timestamp) = ?
            ORDER BY u.user_id, a.timestamp
        ''', (date_str,))
        
        records = cursor.fetchall()
        conn.close()
        
        # Process records by user
        user_data = {}
        for record in records:
            user_id, name, emp_id, dept, punch_type, timestamp = record
            
            if user_id not in user_data:
                user_data[user_id] = {
                    'name': name,
                    'employee_id': emp_id,
                    'department': dept,
                    'punch_in': None,
                    'punch_out': None,
                    'total_hours': 0
                }
            
            if punch_type == 'IN' and user_data[user_id]['punch_in'] is None:
                user_data[user_id]['punch_in'] = timestamp
            elif punch_type == 'OUT':
                user_data[user_id]['punch_out'] = timestamp
        
        # Calculate work hours
        report_data = []
        for user_id, data in user_data.items():
            if data['punch_in'] and data['punch_out']:
                in_time = datetime.strptime(data['punch_in'], '%Y-%m-%d %H:%M:%S')
                out_time = datetime.strptime(data['punch_out'], '%Y-%m-%d %H:%M:%S')
                duration = out_time - in_time
                hours = duration.total_seconds() / 3600
                data['total_hours'] = hours
                status = 'COMPLETE'
            elif data['punch_in']:
                status = 'ACTIVE'
            else:
                status = 'INCOMPLETE'
            
            report_data.append({
                'name': data['name'],
                'employee_id': data['employee_id'],
                'department': data['department'],
                'punch_in': data['punch_in'],
                'punch_out': data['punch_out'],
                'total_hours': data['total_hours'],
                'status': status
            })
        
        return report_data