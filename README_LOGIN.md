# InferIQ - Login System

This document explains the login system for the InferIQ application (formerly known as Deductive Reasoning Framework).

## Overview

The login system is designed to be simple yet secure, using a JSON file to store user credentials instead of a database. The system includes:

- User authentication with bcrypt password hashing
- Role-based access control (Admin and User roles)
- Self-registration option for new users
- User management interface for admins
- Password reset functionality
- First-login password change requirement
- Import/export users via CSV

## Default Login

The system creates a default admin user if no users exist:
- Username: `admin`
- Password: `admin`

You will be prompted to change this password on first login.

## User Management

### For Admins

As an admin, you have access to the User Management page where you can:
1. View all users in the system
2. Create new users
3. Reset passwords for existing users
4. Delete users
5. Export users to CSV
6. Import users from CSV

### Self-Registration

New users can self-register through the Register tab on the login page. This feature can be:
- Enabled/disabled by setting the `ENABLE_REGISTRATION` environment variable to "true" or "false"
- Protected with a registration code by setting `REQUIRE_REGISTRATION_CODE` to "true" and setting a code with `REGISTRATION_CODE`

## Configuration

The login system can be configured using environment variables in the `.env` file:

```
# Storage path for all file storage
INFERIQ_STORAGE=/data/storage

# Enable or disable self-registration (default: true)
ENABLE_REGISTRATION=true

# Require a registration code for self-registration (default: false)
REQUIRE_REGISTRATION_CODE=false

# Set the registration code if required
REGISTRATION_CODE=your_secret_code
```

### Storage Configuration

The `INFERIQ_STORAGE` environment variable defines where all application data is stored, including:
- User credentials (users.json)
- Uploaded files
- Data management files
- Trained models

By default, if this variable is not set, the application will store files in the application directory.

## User Data Storage

User data is stored in `users.json` in the storage directory defined by `INFERIQ_STORAGE`. The file structure is:

```json
{
  "username": {
    "password": "bcrypt_hashed_password",
    "role": "Admin or User",
    "first_login": true/false
  },
  ...
}
```

If the storage directory doesn't exist, it will be created automatically when needed.

## Security Considerations

1. Passwords are hashed using bcrypt
2. The system enforces password confirmation for all password-related operations
3. Only admins can manage other users
4. Users are forced to change their password on first login

## Adding New Users Without Admin Access

If you need to add users without admin access, you can:

1. Temporarily enable self-registration by setting `ENABLE_REGISTRATION=true`
2. Manually edit the `users.json` file (make sure to hash passwords with bcrypt)
3. Use the CSV import feature if you have admin access

## Troubleshooting

If you lose admin access or have issues logging in:
1. Stop the application
2. Run the reset_admin.py script to reset the admin user:
   ```
   python deductive_reasoning_framework/reset_admin.py
   ```
3. This will create a fresh admin user with username "admin" and password "admin" in the storage location defined by `INFERIQ_STORAGE`
4. Restart the application and log in with these credentials
5. You will be prompted to change the password on first login

## About InferIQ

InferIQ is an intelligent inference and reasoning engine that provides:
- Data and rule management
- Inference and validation
- Hyper-parameter tuning
- User management

The application uses a combination of deductive reasoning, Bayesian networks, probabilistic programming, and machine learning to provide accurate inferences with confidence scores.