from app import app, db, User, bcrypt

def setup_initial_data():
    with app.app_context():
        # Drop and Re-create all tables (Warning: This deletes existing data)
        # db.drop_all() # Commented out to prevent accidental data loss
        db.create_all()
        
        # Check if admin already exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            # Create default Admin
            hashed_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
            new_admin = User(
                username='admin',
                password_hash=hashed_password,
                role='admin',
                full_name='System Administrator'
            )
            db.session.add(new_admin)
            
            # Create a sample Faculty
            faculty_pass = bcrypt.generate_password_hash('faculty123').decode('utf-8')
            new_faculty = User(
                username='faculty_user',
                password_hash=faculty_pass,
                role='faculty',
                full_name='Prof. Jane Smith'
            )
            db.session.add(new_faculty)
            
            db.session.commit()
            print("Successfully created default users!")
            print("--------------------------------")
            print("Admin Login:")
            print("Username: admin")
            print("Password: admin123")
            print("--------------------------------")
            print("Faculty Login:")
            print("Username: faculty_user")
            print("Password: faculty123")
        else:
            print("Default users already exist.")

if __name__ == '__main__':
    setup_initial_data()
