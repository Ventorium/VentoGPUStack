//! 内嵌的 Web UI（单文件，shadcn/ui 黑白风格）与图标资源。

pub const INDEX_HTML: &str = include_str!("../../web/index.html");

pub const LOGO_PNG: &[u8] = include_bytes!("../../web/logo.png");
pub const FAVICON_PNG: &[u8] = include_bytes!("../../web/favicon.png");
pub const APPLE_TOUCH_ICON_PNG: &[u8] = include_bytes!("../../web/apple-touch-icon.png");
