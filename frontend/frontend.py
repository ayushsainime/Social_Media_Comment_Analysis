# Reflex frontend application — Social Media Sentiment Analyzer
# Designed by Ayush Saini

from datetime import datetime
from typing import Dict, List

import reflex as rx
import requests

API_URL = "http://localhost:8000"

# ─── Global colour palette ────────────────────────────────────────────
GRADIENT_BG = "url('https://huggingface.co/datasets/ayushsainime/social_media_sentiment_analyzer_media/resolve/main/background_lol_2.jpg') center/cover no-repeat fixed"
CARD_BG = "radial-gradient(circle, rgba(238,174,202,1) 0%, rgba(148,187,233,1) 100%)"
CARD_BORDER = "1px solid rgba(148,187,233,0.5)"
CARD_SHADOW = "0 8px 32px rgba(0,0,0,0.18)"
BLUR = ""
ACCENT_BLUE = "#2A7B9B"
ACCENT_PURPLE = "#7c3aed"
ACCENT_TEAL = "#0d9488"
ACCENT_PINK = "#db2777"
TEXT_PRIMARY = "#1e293b"
TEXT_SECONDARY = "#475569"
POSITIVE_COLOR = "#16a34a"
NEGATIVE_COLOR = "#dc2626"
NEUTRAL_COLOR = "#ca8a04"

FONT_HEADING = "'Courier New', Courier, monospace"
FONT_BODY = "'Courier New', Courier, monospace"


# ─── State ─────────────────────────────────────────────────────────────
class SentimentState(rx.State):
    """State management for sentiment analysis app."""

    # Input values
    text: str = ""
    selected_model: str = "svm"

    # Single prediction output
    result: str = ""
    confidence: int = 0
    confidence_label: str = ""
    error: str = ""
    is_loading: bool = False

    # Model comparison results
    comparison_results: List[Dict] = []
    show_comparison: bool = False

    # Word cloud
    wordcloud_image: str = ""
    show_wordcloud: bool = False

    # Prediction history (last 10)
    history: List[Dict] = []
    has_history: bool = False
    max_history: int = 10

    # Available models (fetched from API)
    available_models: List[str] = []

    # Computed helpers for single-result colour
    result_color: str = NEUTRAL_COLOR

    def load_models(self):
        """Fetch available models when the page loads."""
        try:
            response = requests.get(f"{API_URL}/models")
            if response.status_code == 200:
                self.available_models = response.json()["models"]
                if self.available_models and self.selected_model not in self.available_models:
                    self.selected_model = self.available_models[0]
        except Exception as e:
            self.error = f"Failed to fetch models: {str(e)}"

    def predict_sentiment(self):
        """Call FastAPI backend to predict sentiment with single model."""
        if not self.text.strip():
            self.error = "Please enter some text"
            return

        self.is_loading = True
        self.error = ""
        self.result = ""
        self.confidence = 0
        self.confidence_label = ""
        self.comparison_results = []
        self.show_comparison = False

        try:
            payload = {
                "model_name": self.selected_model,
                "text": self.text,
            }
            response = requests.post(f"{API_URL}/predict", json=payload)

            if response.status_code == 200:
                data = response.json()
                self.result = data["prediction"].upper()
                self.confidence = int(round(data.get("confidence", 0.0) or 0.0))
                self.confidence_label = f"{self.confidence:.1f}%"
                self.result_color = (
                    POSITIVE_COLOR
                    if self.result == "POSITIVE"
                    else NEGATIVE_COLOR
                    if self.result == "NEGATIVE"
                    else NEUTRAL_COLOR
                )

                # Add to history
                self._add_to_history(self.selected_model, self.result, self.confidence)
            else:
                self.error = f"Error: {response.json().get('detail', 'Unknown error')}"
        except Exception as e:
            self.error = f"Connection error: {str(e)}"
        finally:
            self.is_loading = False

    def predict_all_models(self):
        """Call FastAPI backend to predict sentiment with all models."""
        if not self.text.strip():
            self.error = "Please enter some text"
            return

        self.is_loading = True
        self.error = ""
        self.result = ""
        self.confidence = 0
        self.confidence_label = ""
        self.show_comparison = True
        self.comparison_results = []

        try:
            payload = {
                "model_name": "svm",
                "text": self.text,
            }
            response = requests.post(f"{API_URL}/predict-all", json=payload)

            if response.status_code == 200:
                data = response.json()
                self.comparison_results = []
                for res in data["results"]:
                    confidence = int(round(res.get("confidence", 0.0) or 0.0))
                    prediction = res["prediction"].upper()
                    self.comparison_results.append(
                        {
                            "model_name": res["model_name"],
                            "prediction": prediction,
                            "confidence": confidence,
                            "confidence_label": f"{confidence:.1f}%",
                            "color": (
                                POSITIVE_COLOR
                                if prediction == "POSITIVE"
                                else NEGATIVE_COLOR
                                if prediction == "NEGATIVE"
                                else NEUTRAL_COLOR
                            ),
                        }
                    )
            else:
                self.error = f"Error: {response.json().get('detail', 'Unknown error')}"
                self.show_comparison = False
        except Exception as e:
            self.error = f"Connection error: {str(e)}"
            self.show_comparison = False
        finally:
            self.is_loading = False

    def generate_wordcloud(self):
        """Generate word cloud from input text."""
        if not self.text.strip():
            self.error = "Please enter some text"
            return

        self.is_loading = True
        self.error = ""

        try:
            payload = {"text": self.text}
            response = requests.post(f"{API_URL}/wordcloud", json=payload)

            if response.status_code == 200:
                data = response.json()
                self.wordcloud_image = data["image"]
                self.show_wordcloud = True
            else:
                self.error = f"Error: {response.json().get('detail', 'Unknown error')}"
        except Exception as e:
            self.error = f"Connection error: {str(e)}"
        finally:
            self.is_loading = False

    def _add_to_history(self, model: str, prediction: str, confidence: int):
        """Add prediction to history."""
        entry = {
            "text": self.text[:50] + "..." if len(self.text) > 50 else self.text,
            "model": model,
            "prediction": prediction,
            "confidence": confidence,
            "confidence_label": f"{confidence:.1f}%",
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "color": (
                POSITIVE_COLOR
                if prediction == "POSITIVE"
                else NEGATIVE_COLOR
                if prediction == "NEGATIVE"
                else NEUTRAL_COLOR
            ),
        }
        self.history.insert(0, entry)
        if len(self.history) > self.max_history:
            self.history = self.history[: self.max_history]
        self.has_history = len(self.history) > 0

    # Example texts for dropdown
    example_options: List[str] = [
        "😐 Neutral Sentiment",
        "😊 Positive Sentiment",
        "😞 Negative Sentiment",
    ]

    _example_texts: Dict[str, str] = {}

    def _init_examples(self):
        """Build example text lookup."""
        self._example_texts = {
            "😐 Neutral Sentiment": (
                "The weather forecast predicts a mix of sun and clouds for the upcoming week. "
                "Maintenance crews are scheduled to arrive at the office building early tomorrow morning. "
                "Please ensure all documents are submitted through the portal before the deadline passes. "
                "The cafeteria serves a variety of hot and cold meal options every Tuesday afternoon."
            ),
            "😊 Positive Sentiment": (
                "I am absolutely thrilled with the incredible progress our team has achieved this month! "
                "The user interface is exceptionally intuitive and makes the entire experience quite delightful. "
                "Every interaction with the customer support staff has been helpful, kind, and efficient. "
                "We are so grateful for this wonderful opportunity to grow and succeed together."
            ),
            "😞 Negative Sentiment": (
                "I am extremely disappointed that the latest update has caused so many system crashes. "
                "The current navigation is frustratingly slow and often fails to load the requested page. "
                "It is unacceptable that my urgent support tickets have remained unanswered for three days. "
                "This entire process has been a total waste of time and needs immediate improvement."
            ),
        }

    def load_example(self, selected: str):
        """Load an example text into the text area."""
        self._init_examples()
        if selected in self._example_texts:
            self.text = self._example_texts[selected]

    def clear_history(self):
        """Clear prediction history."""
        self.history = []
        self.has_history = False


# ─── Helpers ───────────────────────────────────────────────────────────
def _glass_card(*children, **kwargs) -> rx.Component:
    """Reusable glassmorphism card."""
    return rx.box(
        *children,
        background=CARD_BG,
        border_radius="16px",
        border=CARD_BORDER,
        box_shadow=CARD_SHADOW,
        backdrop_filter=BLUR,
        padding="24px",
        width="100%",
        **kwargs,
    )


def _section_label(text: str, color: str) -> rx.Component:
    """Colored-dot section heading."""
    return rx.hstack(
        rx.box(width="8px", height="8px", border_radius="50%", background=color),
        rx.text(
            text,
            font_family=FONT_BODY,
            font_size="14px",
            font_weight="bold",
            color=TEXT_PRIMARY,
            letter_spacing="0.5px",
        ),
        spacing="2",
        align="center",
    )


# ─── Sections ──────────────────────────────────────────────────────────

def _header() -> rx.Component:
    """Hero header with animated logo and title."""
    return rx.box(
        rx.hstack(
            # Project logo image
            rx.box(
                rx.image(
                    src="https://huggingface.co/datasets/ayushsainime/social_media_sentiment_analyzer_media/resolve/main/social%20media%20sentiment%20analyzer.png",
                    width="100px",
                    height="100px",
                    border_radius="16px",
                    object_fit="cover",
                ),
                flex_shrink="0",
            ),
            # Title + subtitle
            rx.vstack(
                rx.heading(
                    "SOCIAL MEDIA SENTIMENT ANALYZER",
                    font_family=FONT_HEADING,
                    font_size="38px",
                    font_weight="bold",
                    text_align="left",
                    color="#1a1a2e",
                    style={
                        "textShadow": "2px 2px 8px rgba(0,0,0,0.15)",
                    },
                ),
                rx.text(
                    "Powered by Machine Learning  ·  Real-time Analysis",
                    font_family=FONT_BODY,
                    font_size="14px",
                    color=TEXT_SECONDARY,
                    text_align="left",
                    letter_spacing="1px",
                ),
                spacing="2",
                align="start",
                width="100%",
            ),
            spacing="5",
            align="center",
            width="100%",
        ),
        background="rgba(255,255,255,0.85)",
        border_radius="16px",
        padding="20px 24px",
        box_shadow="0 4px 20px rgba(0,0,0,0.12)",
        width="100%",
    )


def _input_section() -> rx.Component:
    """Model selector + text area."""
    return _glass_card(
        # ── Model selector ──
        rx.vstack(
            _section_label("SELECT MODEL", ACCENT_BLUE),
            rx.select(
                items=SentimentState.available_models,
                value=SentimentState.selected_model,
                on_change=SentimentState.set_selected_model,
                placeholder="Choose a model...",
                width="100%",
                size="3",
                style={
                    "fontFamily": FONT_BODY,
                    "backgroundColor": "rgba(255,255,255,0.75)",
                    "border": "1px solid rgba(42,123,155,0.25)",
                    "color": TEXT_PRIMARY,
                    "borderRadius": "12px",
                    "transition": "all 0.3s ease",
                },
            ),
            width="100%",
            spacing="2",
        ),
        rx.box(height="16px"),
        # ── Text area ──
        rx.vstack(
            _section_label("ENTER TEXT", ACCENT_PURPLE),
            rx.text_area(
                value=SentimentState.text,
                on_change=SentimentState.set_text,
                placeholder="Type or paste your text here for sentiment analysis...",
                rows="10",
                width="100%",
                style={
                    "fontFamily": FONT_BODY,
                    "fontSize": "14px",
                    "backgroundColor": "rgba(255,255,255,0.75)",
                    "border": "1px solid rgba(42,123,155,0.25)",
                    "color": TEXT_PRIMARY,
                    "borderRadius": "12px",
                    "transition": "all 0.3s ease",
                    "resize": "vertical",
                    "minHeight": "180px",
                },
            ),
            width="100%",
            spacing="2",
        ),
        rx.box(height="16px"),
        # ── Examples dropdown ──
        rx.vstack(
            _section_label("TRY AN EXAMPLE", ACCENT_TEAL),
            rx.select(
                items=SentimentState.example_options,
                placeholder="Pick an example to load...",
                width="100%",
                size="3",
                on_change=SentimentState.load_example,
                style={
                    "fontFamily": FONT_BODY,
                    "backgroundColor": "rgba(255,255,255,0.75)",
                    "border": "1px solid rgba(13,148,136,0.25)",
                    "color": TEXT_PRIMARY,
                    "borderRadius": "12px",
                    "transition": "all 0.3s ease",
                },
            ),
            width="100%",
            spacing="2",
        ),
    )


def _action_buttons() -> rx.Component:
    """Row of action buttons."""
    return rx.hstack(
        # Analyze
        rx.button(
            rx.hstack(
                rx.html("<span style='font-size:16px;'>⚡</span>"),
                rx.text("Analyze", font_family=FONT_BODY),
                spacing="2",
                align="center",
            ),
            on_click=SentimentState.predict_sentiment,
            loading=SentimentState.is_loading,
            style={
                "background": "linear-gradient(135deg, #6c63ff, #a855f7)",
                "border": "none",
                "color": "white",
                "borderRadius": "12px",
                "fontFamily": FONT_BODY,
                "boxShadow": "0 4px 20px rgba(108,99,255,0.4)",
                "transition": "all 0.3s ease",
                "cursor": "pointer",
            },
            size="3",
        ),
        # Compare All
        rx.button(
            rx.hstack(
                rx.html("<span style='font-size:16px;'>📊</span>"),
                rx.text("Compare All", font_family=FONT_BODY),
                spacing="2",
                align="center",
            ),
            on_click=SentimentState.predict_all_models,
            loading=SentimentState.is_loading,
            variant="outline",
            style={
                "borderColor": ACCENT_TEAL,
                "color": ACCENT_TEAL,
                "borderRadius": "12px",
                "fontFamily": FONT_BODY,
                "transition": "all 0.3s ease",
                "cursor": "pointer",
            },
            size="3",
        ),
        # Word Cloud
        rx.button(
            rx.hstack(
                rx.html("<span style='font-size:16px;'>☁️</span>"),
                rx.text("Word Cloud", font_family=FONT_BODY),
                spacing="2",
                align="center",
            ),
            on_click=SentimentState.generate_wordcloud,
            loading=SentimentState.is_loading,
            variant="outline",
            style={
                "borderColor": ACCENT_PINK,
                "color": ACCENT_PINK,
                "borderRadius": "12px",
                "fontFamily": FONT_BODY,
                "transition": "all 0.3s ease",
                "cursor": "pointer",
            },
            size="3",
        ),
        spacing="3",
        wrap="wrap",
        width="100%",
        justify="center",
    )


def _error_banner() -> rx.Component:
    """Conditional error alert."""
    return rx.cond(
        SentimentState.error != "",
        rx.box(
            rx.hstack(
                rx.html("<span style='font-size:18px;'>⚠️</span>"),
                rx.text(
                    SentimentState.error,
                    font_family=FONT_BODY,
                    font_size="14px",
                    color=NEGATIVE_COLOR,
                ),
                spacing="2",
                align="center",
            ),
            width="100%",
            padding="12px 16px",
            border_radius="12px",
            background="rgba(248,113,113,0.1)",
            border="1px solid rgba(248,113,113,0.3)",
        ),
    )


def _single_result() -> rx.Component:
    """Result card for single-model prediction."""
    return rx.cond(
        SentimentState.result != "",
        _glass_card(
            _section_label("PREDICTION RESULT", ACCENT_TEAL),
            rx.box(height="12px"),
            rx.hstack(
                rx.vstack(
                    rx.text(
                        SentimentState.result,
                        font_family=FONT_BODY,
                        font_size="28px",
                        font_weight="bold",
                        color=SentimentState.result_color,
                    ),
                    rx.text(
                        SentimentState.confidence_label + " Confidence",
                        font_family=FONT_BODY,
                        font_size="13px",
                        color=TEXT_SECONDARY,
                    ),
                    spacing="1",
                    align="start",
                ),
                rx.box(
                    rx.cond(
                        SentimentState.result == "POSITIVE",
                        rx.html("<span style='font-size:48px;'>😊</span>"),
                        rx.cond(
                            SentimentState.result == "NEGATIVE",
                            rx.html("<span style='font-size:48px;'>😞</span>"),
                            rx.html("<span style='font-size:48px;'>😐</span>"),
                        ),
                    ),
                    margin_left="auto",
                ),
                width="100%",
                align="center",
                justify="start",
            ),
            rx.box(height="12px"),
            # Confidence bar
            rx.hstack(
                rx.box(
                    rx.progress(
                        value=SentimentState.confidence,
                        max=100,
                        width="100%",
                        style={
                            "borderRadius": "9999px",
                            "overflow": "hidden",
                            "height": "12px",
                        },
                    ),
                    width="100%",
                    border_radius="9999px",
                    overflow="hidden",
                    background="rgba(255,255,255,0.45)",
                ),
                rx.text(
                    SentimentState.confidence_label,
                    font_family=FONT_BODY,
                    font_size="13px",
                    color=TEXT_PRIMARY,
                    min_width="50px",
                    text_align="right",
                ),
                width="100%",
                spacing="3",
                align="center",
            ),
        ),
    )


def _comparison_section() -> rx.Component:
    """Model comparison table with aligned confidence bars."""
    return rx.cond(
        SentimentState.show_comparison,
        _glass_card(
            _section_label("MODEL COMPARISON", ACCENT_TEAL),
            rx.box(height="16px"),
            # Column header
            rx.hstack(
                rx.text("Model", font_family=FONT_BODY, font_size="11px",
                        color=TEXT_SECONDARY, width="140px",
                        text_transform="uppercase", letter_spacing="1px"),
                rx.text("Sentiment", font_family=FONT_BODY, font_size="11px",
                        color=TEXT_SECONDARY, width="100px",
                        text_transform="uppercase", letter_spacing="1px"),
                rx.text("Confidence", font_family=FONT_BODY, font_size="11px",
                        color=TEXT_SECONDARY, text_transform="uppercase",
                        letter_spacing="1px", flex="1"),
                width="100%",
                padding_bottom="8px",
                border_bottom="1px solid rgba(30,41,59,0.12)",
            ),
            rx.box(height="8px"),
            rx.foreach(
                SentimentState.comparison_results,
                lambda r: rx.vstack(
                    rx.hstack(
                        # Model name
                        rx.text(
                            r["model_name"],
                            font_family=FONT_BODY,
                            font_size="13px",
                            color=TEXT_PRIMARY,
                            width="140px",
                            font_weight="bold",
                        ),
                        # Sentiment badge
                        rx.box(
                            rx.text(
                                r["prediction"],
                                font_family=FONT_BODY,
                                font_size="12px",
                                font_weight="bold",
                                color=r["color"],
                            ),
                            padding="4px 12px",
                            border_radius="9999px",
                            background=rx.cond(
                                r["prediction"] == "POSITIVE",
                                "rgba(52,211,153,0.12)",
                                rx.cond(
                                    r["prediction"] == "NEGATIVE",
                                    "rgba(248,113,113,0.12)",
                                    "rgba(251,191,36,0.12)",
                                ),
                            ),
                            border=rx.cond(
                                r["prediction"] == "POSITIVE",
                                "1px solid rgba(52,211,153,0.3)",
                                rx.cond(
                                    r["prediction"] == "NEGATIVE",
                                    "1px solid rgba(248,113,113,0.3)",
                                    "1px solid rgba(251,191,36,0.3)",
                                ),
                            ),
                            width="100px",
                            text_align="center",
                        ),
                        # Confidence bar + label
                        rx.hstack(
                            rx.box(
                                rx.progress(
                                    value=r["confidence"],
                                    max=100,
                                    width="100%",
                                    style={
                                        "borderRadius": "9999px",
                                        "overflow": "hidden",
                                        "height": "10px",
                                    },
                                ),
                                width="100%",
                                border_radius="9999px",
                                overflow="hidden",
                                background="rgba(255,255,255,0.45)",
                            ),
                            rx.text(
                                r["confidence_label"],
                                font_family=FONT_BODY,
                                font_size="12px",
                                color=TEXT_PRIMARY,
                                min_width="45px",
                                text_align="right",
                            ),
                            flex="1",
                            spacing="3",
                            align="center",
                        ),
                        width="100%",
                        align="center",
                    ),
                    rx.box(width="100%", height="1px",
                           background="rgba(30,41,59,0.08)"),
                    width="100%",
                    spacing="2",
                    padding_y="6px",
                ),
            ),
        ),
    )


def _wordcloud_section() -> rx.Component:
    """Word cloud display."""
    return rx.cond(
        SentimentState.show_wordcloud,
        _glass_card(
            _section_label("WORD CLOUD", ACCENT_PINK),
            rx.box(height="12px"),
            rx.box(
                rx.image(
                    src="data:image/png;base64," + SentimentState.wordcloud_image,
                    width="100%",
                    max_width="700px",
                    border_radius="12px",
                ),
                padding="8px",
                border_radius="12px",
                background="rgba(255,255,255,0.5)",
            ),
            align="center",
        ),
    )


def _history_section() -> rx.Component:
    """Prediction history panel."""
    return rx.cond(
        SentimentState.has_history,
        _glass_card(
            rx.hstack(
                _section_label("RECENT PREDICTIONS", ACCENT_PINK),
                rx.button(
                    rx.text("Clear All", font_family=FONT_BODY, font_size="12px"),
                    on_click=SentimentState.clear_history,
                    variant="ghost",
                    size="1",
                    style={
                        "color": NEGATIVE_COLOR,
                        "fontFamily": FONT_BODY,
                        "borderRadius": "8px",
                    },
                ),
                width="100%",
                justify="between",
            ),
            rx.box(height="12px"),
            # Header row
            rx.hstack(
                rx.text("Time", font_family=FONT_BODY, font_size="11px",
                        color=TEXT_SECONDARY, width="60px",
                        text_transform="uppercase", letter_spacing="1px"),
                rx.text("Text", font_family=FONT_BODY, font_size="11px",
                        color=TEXT_SECONDARY, width="180px",
                        text_transform="uppercase", letter_spacing="1px"),
                rx.text("Model", font_family=FONT_BODY, font_size="11px",
                        color=TEXT_SECONDARY, width="100px",
                        text_transform="uppercase", letter_spacing="1px"),
                rx.text("Result", font_family=FONT_BODY, font_size="11px",
                        color=TEXT_SECONDARY,
                        text_transform="uppercase", letter_spacing="1px"),
                width="100%",
                padding_bottom="8px",
                border_bottom="1px solid rgba(30,41,59,0.12)",
            ),
            rx.box(height="4px"),
            rx.foreach(
                SentimentState.history,
                lambda item: rx.vstack(
                    rx.hstack(
                        rx.text(item["timestamp"], font_family=FONT_BODY,
                                font_size="12px", color=TEXT_SECONDARY, width="60px"),
                        rx.text(item["text"], font_family=FONT_BODY,
                                font_size="12px", color=TEXT_PRIMARY, width="180px",
                                overflow="hidden", text_overflow="ellipsis",
                                white_space="nowrap"),
                        rx.text(item["model"], font_family=FONT_BODY,
                                font_size="12px", color=ACCENT_BLUE, width="100px"),
                        rx.box(
                            rx.text(
                                item["prediction"],
                                font_family=FONT_BODY,
                                font_size="11px",
                                font_weight="bold",
                                color=item["color"],
                            ),
                            padding="3px 10px",
                            border_radius="9999px",
                            background=rx.cond(
                                item["prediction"] == "POSITIVE",
                                "rgba(52,211,153,0.12)",
                                rx.cond(
                                    item["prediction"] == "NEGATIVE",
                                    "rgba(248,113,113,0.12)",
                                    "rgba(251,191,36,0.12)",
                                ),
                            ),
                            border=rx.cond(
                                item["prediction"] == "POSITIVE",
                                "1px solid rgba(52,211,153,0.3)",
                                rx.cond(
                                    item["prediction"] == "NEGATIVE",
                                    "1px solid rgba(248,113,113,0.3)",
                                    "1px solid rgba(251,191,36,0.3)",
                                ),
                            ),
                        ),
                        width="100%",
                        spacing="2",
                        align="center",
                    ),
                    rx.box(width="100%", height="1px",
                           background="rgba(30,41,59,0.08)"),
                    width="100%",
                    spacing="2",
                    padding_y="6px",
                ),
            ),
        ),
    )


def _models_footer() -> rx.Component:
    """Available models strip."""
    return rx.box(
        rx.hstack(
            rx.text(
                "AVAILABLE MODELS:",
                font_family=FONT_BODY,
                font_size="12px",
                color=TEXT_SECONDARY,
                font_weight="bold",
            ),
            rx.foreach(
                SentimentState.available_models,
                lambda model: rx.box(
                    rx.text(model, font_family=FONT_BODY, font_size="11px",
                            color=ACCENT_BLUE),
                    padding="4px 12px",
                    border_radius="9999px",
                    background="rgba(42,123,155,0.1)",
                    border="1px solid rgba(42,123,155,0.25)",
                ),
            ),
            spacing="2",
            wrap="wrap",
            width="100%",
            align="center",
        ),
        background="rgba(255,255,255,0.85)",
        border_radius="12px",
        padding="12px 16px",
        box_shadow="0 2px 12px rgba(0,0,0,0.08)",
        width="100%",
    )


def _credits() -> rx.Component:
    """Credits section."""
    return rx.box(
        rx.vstack(
            rx.box(
                width="60px",
                height="1px",
                background="linear-gradient(90deg, transparent, rgba(30,41,59,0.2), transparent)",
            ),
            rx.hstack(
                rx.text("Crafted with ", font_family=FONT_BODY, font_size="13px",
                        color=TEXT_SECONDARY),
                rx.html("<span style='font-size:14px;'>💜</span>"),
                rx.text(" by ", font_family=FONT_BODY, font_size="13px",
                        color=TEXT_SECONDARY),
                rx.link(
                    rx.text("Ayush Saini", font_family=FONT_BODY, font_size="13px",
                            color=ACCENT_BLUE, font_weight="bold"),
                    href="https://www.linkedin.com/in/ayush-saini-30a4a0372/",
                    is_external=True,
                    style={"textDecoration": "none"},
                ),
                spacing="0",
                align="center",
                justify="center",
            ),
            rx.link(
                rx.hstack(
                    rx.html(
                        "<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' "
                        "viewBox='0 0 24 24' fill='none' stroke='#6c63ff' stroke-width='2' "
                        "stroke-linecap='round' stroke-linejoin='round'>"
                        "<path d='M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z'/>"
                        "<rect x='2' y='9' width='4' height='12'/><circle cx='4' cy='4' r='2'/></svg>"
                    ),
                    rx.text("LinkedIn", font_family=FONT_BODY, font_size="12px",
                            color=ACCENT_BLUE),
                    spacing="1",
                    align="center",
                ),
                href="https://www.linkedin.com/in/ayush-saini-30a4a0372/",
                is_external=True,
                style={"textDecoration": "none"},
            ),
            spacing="3",
            align="center",
            width="100%",
            padding_top="4px",
        ),
        background="rgba(255,255,255,0.85)",
        border_radius="12px",
        padding="12px 16px",
        box_shadow="0 2px 12px rgba(0,0,0,0.08)",
        width="100%",
    )


# ─── Page ──────────────────────────────────────────────────────────────
def index() -> rx.Component:
    """Main page — two-column block layout."""
    return rx.box(
        rx.hstack(
            # ━━━ LEFT COLUMN ━━━
            rx.vstack(
                _header(),
                rx.box(height="8px"),
                _input_section(),
                rx.box(height="12px"),
                _models_footer(),
                width="100%",
                align="center",
                flex="1",
            ),
            # ━━━ RIGHT COLUMN ━━━
            rx.vstack(
                _action_buttons(),
                rx.box(height="8px"),
                _error_banner(),
                rx.box(height="8px"),
                _single_result(),
                rx.box(height="10px"),
                _comparison_section(),
                rx.box(height="10px"),
                _wordcloud_section(),
                rx.box(height="10px"),
                _history_section(),
                rx.box(height="20px"),
                rx.spacer(),
                _credits(),
                width="100%",
                align="center",
                flex="1",
                min_height="80vh",
            ),
            width="100%",
            spacing="6",
            align="start",
        ),
        width="100%",
        min_height="100vh",
        style={
            "background": GRADIENT_BG,
            "fontFamily": FONT_BODY,
            "color": TEXT_PRIMARY,
            "display": "flex",
            "justifyContent": "center",
            "alignItems": "flex-start",
        },
        padding="24px 16px",
    )


app = rx.App(
    style={
        "background": GRADIENT_BG,
        "font_family": FONT_BODY,
    },
)
app.add_page(
    index,
    on_load=SentimentState.load_models,
    title="Social Media Sentiment Analyzer",
)