"""
Authentication module for Cliplink Backend

This module is a placeholder for future authentication implementation.
Currently disabled - all endpoints are open.
"""

from fastapi import Depends
from typing import Optional
from pydantic import BaseModel


class User(BaseModel):
    """User model representing authenticated user data"""
    id: str = "anonymous"
    email: str = "anonymous@example.com"
    name: Optional[str] = "Anonymous User"
    role: str = "user"


async def get_current_user() -> User:
    """
    Placeholder dependency that returns an anonymous user
    
    Replace this with your actual authentication logic when ready.
    """
    return User()


async def get_optional_user() -> Optional[User]:
    """
    Placeholder dependency for optional authentication
    
    Returns anonymous user for now.
    """
    return User()


def require_role(required_role: str):
    """
    Placeholder role checker - currently allows all access
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        return user
    
    return role_checker


# Convenience dependency for admin-only endpoints
get_admin_user = require_role("admin") 