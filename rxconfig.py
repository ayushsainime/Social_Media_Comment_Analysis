import reflex as rx
from reflex.plugins.sitemap import SitemapPlugin

config = rx.Config(
    app_name="frontend",
    title="Sentiment Analysis App",
    backend_port=8001,
    frontend_port=3000,
    frontend_host="0.0.0.0",
    disable_plugins=[SitemapPlugin],
)