"""
Authentication module for admin access
"""

import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

# Admin configuration
ADMIN_PASSWORD = "hegopinath"
active_tokens: Dict[str, Dict[str, Any]] = {}  # In production, use Redis or database

# Security scheme
security = HTTPBearer(auto_error=False)

class LoginRequest(BaseModel):
    password: str

class LoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: str

def generate_token() -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(32)

def verify_admin_password(password: str) -> bool:
    """Verify admin password"""
    return password == ADMIN_PASSWORD

def create_admin_token() -> str:
    """Create a new admin token"""
    token = generate_token()
    active_tokens[token] = {
        'created_at': datetime.now(),
        'expires_at': datetime.now() + timedelta(hours=24)  # Token expires in 24 hours
    }
    logger.info(f"Created admin token: {token[:8]}...")
    return token

def verify_admin_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify admin token"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    token = credentials.credentials
    if token not in active_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    token_info = active_tokens[token]
    if datetime.now() > token_info['expires_at']:
        del active_tokens[token]
        raise HTTPException(status_code=401, detail="Token expired")
    
    return True

def cleanup_expired_tokens():
    """Clean up expired tokens"""
    current_time = datetime.now()
    expired_tokens = [
        token for token, info in active_tokens.items()
        if current_time > info['expires_at']
    ]
    
    for token in expired_tokens:
        del active_tokens[token]
    
    if expired_tokens:
        logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")

def get_active_tokens_count() -> int:
    """Get count of active tokens"""
    cleanup_expired_tokens()
    return len(active_tokens) 