//! 管理网关：独立的 pingora 代理服务，request_filter 内直接响应，永远不上游。

pub mod api;
pub mod auth;
pub mod web;

use std::sync::Arc;

use async_trait::async_trait;
use pingora::prelude::{HttpPeer, ProxyHttp, Session};
use pingora_error::Result;

use crate::config::AppState;

pub struct AdminCtx;

pub struct AdminGw {
    pub state: Arc<AppState>,
}

#[async_trait]
impl ProxyHttp for AdminGw {
    type CTX = AdminCtx;

    fn new_ctx(&self) -> Self::CTX {
        AdminCtx
    }

    async fn request_filter(&self, session: &mut Session, _ctx: &mut AdminCtx) -> Result<bool> {
        api::handle(session, &self.state).await;
        Ok(true)
    }

    async fn upstream_peer(&self, _session: &mut Session, _ctx: &mut AdminCtx) -> Result<Box<HttpPeer>> {
        Err(pingora_error::Error::new_str("admin gateway: no upstream"))
    }
}
