from __future__ import annotations
"""인증 & RBAC — 시연용 하드코딩 계정 3개
실제 임상 환경에서는 이 부분을 병원 LDAP / AD SSO로 교체
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# 시연용 시크릿 (실제 운영 시 환경변수로 교체)
SECRET_KEY = "hms-eeg-demo-secret-key-replace-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)

# 시연용 계정 (실제 임상: 병원 LDAP/AD SSO로 교체)
DEMO_USERS: dict[str, dict] = {
    "eeg_tech": {
        "username": "eeg_tech",
        "full_name": "EEG 기사 (김기사)",
        "role": "technician",
        "hashed_password": pwd_context.hash("tech1234"),
    },
    "resident": {
        "username": "resident",
        "full_name": "레지던트 (이전공)",
        "role": "resident",
        "hashed_password": pwd_context.hash("res1234"),
    },
    "neurologist": {
        "username": "neurologist",
        "full_name": "신경과 전문의 (박전문)",
        "role": "neurologist",
        "hashed_password": pwd_context.hash("neuro1234"),
    },
}

# 역할별 권한
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "technician":  {"upload", "view_own"},
    "resident":    {"upload", "view_own", "view_history"},
    "neurologist": {"upload", "view_own", "view_history", "view_alerts", "ack_alert"},
}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    full_name: str


class TokenData(BaseModel):
    username: str
    role: str


def authenticate_user(username: str, password: str) -> dict | None:
    user = DEMO_USERS.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return None
    return user


def create_access_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> TokenData:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="인증이 필요합니다. 로그인 후 토큰을 헤더에 포함하세요.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise exc
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        role: str = payload.get("role", "")
        if not username or not role:
            raise exc
    except JWTError:
        raise exc
    return TokenData(username=username, role=role)


def require_permission(permission: str):
    """특정 권한이 있는 역할만 허용하는 Depends 팩토리"""
    async def checker(
        user: Annotated[TokenData, Depends(get_current_user)],
    ) -> TokenData:
        if permission not in ROLE_PERMISSIONS.get(user.role, set()):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"권한 없음 (필요: {permission}, 현재 역할: {user.role})",
            )
        return user
    return checker
