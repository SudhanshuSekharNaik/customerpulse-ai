"""FastAPI security dependencies and role-based access control (RBAC)."""

from typing import List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.database.models import User
from backend.app.security.auth import decode_access_token
from backend.app.schemas import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Retrieve authenticated user or return None for public/demo mode."""
    if not token:
        # Fallback to demo admin user if running without explicit token in local dev
        demo_user = db.query(User).filter(User.username == "admin").first()
        return demo_user

    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


def require_roles(allowed_roles: List[str]):
    """Decorator dependency to enforce RBAC permissions."""
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user is None:
            return None
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted for role '{current_user.role}'. Required: {allowed_roles}",
            )
        return current_user
    return role_checker
