//! 认证：argon2id 密码哈希 + JWT(HS256) 会话令牌

use argon2::password_hash::PasswordHash;
use argon2::{Argon2, PasswordVerifier};
use chrono::Utc;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};

pub fn verify_password(password: &str, hash: &str) -> Result<bool, String> {
    let parsed = PasswordHash::new(hash).map_err(|e| e.to_string())?;
    Ok(Argon2::default()
        .verify_password(password.as_bytes(), &parsed)
        .is_ok())
}

#[derive(Serialize, Deserialize)]
struct Claims {
    sub: String,
    iat: i64,
    exp: i64,
}

pub fn issue_token(secret: &str, user: &str, ttl_hours: u64) -> Result<String, String> {
    let now = Utc::now();
    let claims = Claims {
        sub: user.to_string(),
        iat: now.timestamp(),
        exp: now.timestamp() + (ttl_hours as i64) * 3600,
    };
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .map_err(|e| e.to_string())
}

pub fn verify_token(secret: &str, token: &str) -> Result<String, String> {
    if token.trim().is_empty() {
        return Err("empty token".into());
    }
    let data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(secret.as_bytes()),
        &Validation::default(),
    )
    .map_err(|e| e.to_string())?;
    Ok(data.claims.sub)
}
