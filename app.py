"""
app.py
======

Quantitative Portfolio Optimization Dashboard -- desktop build (PySide6).


- `src.data_engine.DataEngine`: loads/pivots raw historical Close prices
  and converts them to daily log returns via `calculate_log_returns`.
- `src.optimizer.PortfolioOptimizer`: runs Max Sharpe Ratio optimization
  over a user-selected basket of tickers. It expects daily log returns
  as input, not raw price levels.

Structure (top to bottom):
    1. Constants
    2. DataCache          -- manual cache replacing st.cache_resource/cache_data
    3. OptimizationWorker -- QThread pipeline runner (keeps the UI responsive)
    4. ConfigurationPanel -- sidebar tabs: universe, parameters, constraints
    5. MetricCard / DonutChart -- small display widgets
    6. ResultsPanel       -- metrics + chart/holdings tabs + allocation table
    7. MainWindow         -- wires everything together
    8. main()             -- entry point

Run with:
    python app.py
"""

from __future__ import annotations

import datetime as dt
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QDate, QRect, QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.data_engine import DataEngine
from src.optimizer import PortfolioOptimizer

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

DATA_FILE_PATH: str = "data/stock_details_5_years.csv"
DEFAULT_TICKERS: List[str] = ["AAPL", "MSFT", "GOOGL"]
DEFAULT_LOOKBACK_YEARS: int = 5

# Weights at or below this threshold are treated as an effective zero
# allocation. SLSQP can leave tiny floating-point residuals (e.g. 1e-10)
# on bound-constrained assets instead of an exact 0.0, so a strict
# `weight > 0.0` check is not sufficient to keep the chart/table clean.
ZERO_WEIGHT_TOLERANCE: float = 1e-4

# Slider is an integer widget; 400 steps over a 0.0-0.10 range gives a
# 0.00025 resolution, matching the granularity of the original st.slider
# without needing a float-aware Qt control.
RISK_FREE_SLIDER_MAX_STEPS: int = 400
RISK_FREE_MAX_RATE: float = 0.10
RISK_FREE_STEP: float = RISK_FREE_MAX_RATE / RISK_FREE_SLIDER_MAX_STEPS

LOOKBACK_PRESETS: Dict[str, Optional[int]] = {
    "1 Year": 1,
    "3 Years": 3,
    "5 Years": 5,
    "10 Years": 10,
    "Custom": None,
}

ALLOCATION_COLORS: List[str] = [
    "#0b1a2b",
    "#c6a56a",
    "#8c7a66",
    "#d4c4a8",
    "#4a3f2f",
    "#b08d57",
    "#c4b6a6",
    "#6e5b3e",
    "#e0d0b0",
    "#2c3a4d",
]
CHART_SURFACE: str = "#f6ebd8"

PriceCacheKey = Tuple[str, Tuple[str, ...], str, str]

APP_STYLESHEET = """
QMainWindow {
    background-color: #f6ebd8;
    color: #0b1a2b;
    font-size: 13px;
}
QWidget {
    color: #0b1a2b;
    font-size: 13px;
}
QLabel {
    background-color: transparent;
}
QFrame#headerBar {
    background-color: #0b1a2b;
    min-height: 84px;
}
QLabel#appTitle {
    color: #c6a56a;
    font-size: 26px;
    font-weight: 700;
    background-color: transparent;
}
QLabel#appSubtitle {
    color: #f6ebd8;
    font-size: 13px;
    background-color: transparent;
}
QWidget#sidebar {
    background-color: #f3e6d0;
    border-right: 1px solid #c4b6a6;
}
QWidget#windowRoot, QWidget#resultsPanel, QSplitter {
    background-color: #f6ebd8;
}
QTabWidget::pane {
    border: 1px solid #c4b6a6;
    background: #f6ebd8;
    border-radius: 6px;
    top: -1px;
}
QTabBar::tab {
    padding: 8px 14px;
    background: #c4b6a6;
    border: 1px solid #c4b6a6;
    color: #0b1a2b;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #f6ebd8;
    color: #0b1a2b;
    font-weight: 600;
    border-bottom-color: #f6ebd8;
}
QPushButton {
    background-color: #c4b6a6;
    color: #0b1a2b;
    border: 1px solid #b3a494;
    padding: 6px 12px;
    border-radius: 5px;
}
QPushButton:hover {
    background-color: #d2c4b4;
}
QPushButton#generateButton {
    background-color: #c6a56a;
    color: #0b1a2b;
    border: none;
    padding: 10px 16px;
    border-radius: 6px;
    font-weight: 600;
}
QPushButton#generateButton:hover {
    background-color: #d4b57a;
}
QPushButton#generateButton:disabled {
    background-color: #d8c9a8;
    color: #6e5b3e;
}
QFrame#metricCard {
    background-color: #f6ebd8;
    border: 1px solid #c4b6a6;
    border-radius: 8px;
}
QFrame#metricCard QLabel {
    background-color: transparent;
}
QLabel#metricTitle {
    color: #0b1a2b;
    font-size: 12px;
    background-color: transparent;
}
QLabel#metricValue {
    font-size: 24px;
    font-weight: 600;
    color: #0b1a2b;
    background-color: transparent;
}
QLabel#emptyStateTitle {
    color: #0b1a2b;
    font-size: 16px;
    font-weight: 600;
    background-color: transparent;
}
QLabel#emptyStateBody, QLabel#hintLabel, QLabel#countLabel {
    color: #6e5b3e;
    font-size: 12px;
    background-color: transparent;
}
QTableWidget {
    border: 1px solid #c4b6a6;
    gridline-color: #e8dcc8;
    background: #f6ebd8;
    alternate-background-color: #efe0c8;
    selection-background-color: #c6a56a;
    selection-color: #0b1a2b;
}
QHeaderView::section {
    background: #c4b6a6;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #b3a494;
    font-weight: 600;
    color: #0b1a2b;
}
QLineEdit, QComboBox, QDateEdit, QDoubleSpinBox {
    padding: 5px 8px;
    border: 1px solid #c4b6a6;
    border-radius: 4px;
    background: #f6ebd8;
}
QListWidget#tickerGrid {
    background: transparent;
    border: none;
    outline: none;
    padding: 2px;
}
QListWidget#tickerGrid::item {
    background: transparent;
    border: none;
}
QFrame#selectedSummary {
    background-color: #f6ebd8;
    border: 1px solid #c4b6a6;
    border-radius: 4px;
}
QLabel#selectedTickers {
    color: #0b1a2b;
    background-color: transparent;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #c4b6a6;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -6px 0;
    background: #c6a56a;
    border-radius: 7px;
}
QCheckBox {
    spacing: 8px;
}
QLabel#statusWarning {
    color: #8a6a2f;
}
QLabel#statusError {
    color: #8b2e2e;
}
QLabel#statusInfo {
    color: #0b1a2b;
}
"""


# ---------------------------------------------------------------------------
# 2. DataCache -- manual cache replacing st.cache_resource / st.cache_data
# ---------------------------------------------------------------------------


class DataCache:
    """
    Owns a single DataEngine instance and memoizes its expensive calls.

    Streamlit's cache decorators have no desktop equivalent, so this
    reproduces the two caching strategies the dashboard relies on:
    one DataEngine per CSV file path (resource caching), and memoized
    price/returns lookups keyed on the actual call arguments (data
    caching), so repeated identical requests skip disk I/O.
    """

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._engine = DataEngine(file_path)
        self._ticker_cache: Optional[List[str]] = None
        self._price_cache: Dict[PriceCacheKey, pd.DataFrame] = {}
        self._returns_cache: Dict[int, pd.DataFrame] = {}

    def get_available_tickers(self) -> List[str]:
        """Scan the CSV header/ticker column once and cache the result."""
        if self._ticker_cache is not None:
            return self._ticker_cache

        # Priority order matters: DataEngine filters on a column literally
        # named "Company" (DataEngine.COMPANY_COLUMN), so the ticker values
        # shown in the UI must come from that same column or selections
        # will silently match zero rows.
        candidate_columns: Tuple[str, ...] = (
            "Company",
            "Symbol",
            "Ticker",
            "company",
            "symbol",
            "ticker",
        )
        header_columns = pd.read_csv(self._file_path, nrows=0).columns.tolist()
        ticker_column = next(
            (col for col in candidate_columns if col in header_columns), None
        )
        if ticker_column is None:
            raise ValueError(
                "Could not locate a ticker/symbol column in the dataset. "
                f"Available columns: {header_columns}"
            )

        tickers = (
            pd.read_csv(self._file_path, usecols=[ticker_column])[ticker_column]
            .dropna()
            .unique()
            .tolist()
        )
        self._ticker_cache = sorted(tickers)
        return self._ticker_cache

    def load_price_data(
        self, tickers: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load and pivot Close prices, memoized on (tickers, start, end)."""
        key: PriceCacheKey = (
            self._file_path,
            tuple(sorted(tickers)),
            start_date,
            end_date,
        )
        if key not in self._price_cache:
            self._price_cache[key] = self._engine.load_and_pivot_data(
                tickers, start_date, end_date
            )
        return self._price_cache[key]

    def get_log_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert price levels to daily log returns, memoized by content.

        PortfolioOptimizer expects daily log returns (it internally
        annualizes with mean() * 252 / cov() * 252), so feeding it raw
        prices silently produces wildly inflated return/volatility
        figures -- this conversion is mandatory, not optional.
        """
        cache_key = int(pd.util.hash_pandas_object(prices_df).sum())
        if cache_key not in self._returns_cache:
            self._returns_cache[cache_key] = self._engine.calculate_log_returns(
                prices_df
            )
        return self._returns_cache[cache_key]


# ---------------------------------------------------------------------------
# 3. OptimizationWorker -- QThread pipeline runner
# ---------------------------------------------------------------------------


class OptimizationWorker(QThread):
    """
    Runs the load -> transform -> optimize pipeline off the UI thread.

    Streamlit reruns the whole script top-to-bottom on every interaction,
    which is tolerable because each cached step is cheap on a cache hit.
    A desktop event loop has no such rerun model: any blocking work run
    directly in a slot would freeze the UI. QThread + signals keeps the
    UI responsive while this work happens in the background.
    """

    succeeded = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        data_cache: DataCache,
        tickers: List[str],
        start_date: str,
        end_date: str,
        risk_free_rate: float,
        constraints: Optional[Dict[str, Optional[float]]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._data_cache = data_cache
        self._tickers = tickers
        self._start_date = start_date
        self._end_date = end_date
        self._risk_free_rate = risk_free_rate
        # Optional Generate Portfolio constraints: 'min_return' and
        # 'max_volatility' are each a fraction or None (no constraint);
        # 'min_weight'/'max_weight' are fractions defaulting to the
        # unconstrained (0.0, 1.0) allocation range. Defaults to an empty
        # dict so a worker constructed the old way (no constraints arg)
        # behaves exactly as before.
        self._constraints: Dict[str, Optional[float]] = constraints or {}

    def run(self) -> None:  # noqa: D102 -- Qt override, not a public API method
        try:
            prices_df = self._data_cache.load_price_data(
                self._tickers, self._start_date, self._end_date
            )
            if prices_df.empty:
                self.failed.emit(
                    "No price data found for the selected tickers and date "
                    "range. Try widening the timeframe or choosing different "
                    "assets."
                )
                return

            log_returns_df = self._data_cache.get_log_returns(prices_df)
            if log_returns_df.empty:
                self.failed.emit(
                    "Not enough price history in the selected date range to "
                    "compute returns. Try widening the timeframe."
                )
                return

            optimizer = PortfolioOptimizer(log_returns_df)

            min_return = self._constraints.get("min_return")
            max_volatility = self._constraints.get("max_volatility")
            min_weight = self._constraints.get("min_weight") or 0.0
            max_weight = self._constraints.get("max_weight")
            if max_weight is None:
                max_weight = 1.0

            constraints_active = (
                min_return is not None
                or max_volatility is not None
                or min_weight > 0.0
                or max_weight < 1.0
            )

            if constraints_active:
                result = optimizer.optimize_with_constraints(
                    tickers=self._tickers,
                    risk_free_rate=self._risk_free_rate,
                    min_return=min_return,
                    max_volatility=max_volatility,
                    min_weight=min_weight,
                    max_weight=max_weight,
                )
            else:
                result = optimizer.optimize_max_sharpe(
                    tickers=self._tickers, risk_free_rate=self._risk_free_rate
                )
            self.succeeded.emit(result)
        except ValueError as error:
            self.failed.emit(f"Portfolio optimization failed: {error}")
        except Exception as error:  # noqa: BLE001 -- surface any failure to the UI
            self.failed.emit(f"Unexpected error: {error}")


# ---------------------------------------------------------------------------
# 4. ConfigurationPanel -- left sidebar
# ---------------------------------------------------------------------------

NAVY = QColor("#0b1a2b")
GOLD = QColor("#c6a56a")
CREAM = QColor("#efe0c8")
TAUPE = QColor("#c4b6a6")
CHAMPAGNE = QColor("#f6ebd8")
MUTED = QColor("#8c7a66")


class TickerCardDelegate(QStyledItemDelegate):
    """Paints ticker items as selectable cards instead of checkbox rows."""

    def paint(self, painter: QPainter, option, index) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        rect = option.rect.adjusted(3, 3, -3, -3)
        painter.setBrush(NAVY if selected else CREAM)
        painter.setPen(QPen(GOLD if selected else TAUPE, 2 if selected else 1))
        painter.drawRoundedRect(rect, 8, 8)

        ticker = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        name_font = QFont(option.font)
        name_font.setBold(True)
        painter.setFont(name_font)
        painter.setPen(GOLD if selected else NAVY)
        name_rect = QRect(rect.left(), rect.top() + 6, rect.width(), 20)
        painter.drawText(
            name_rect,
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            ticker,
        )

        status_font = QFont(option.font)
        status_font.setPointSize(max(8, option.font.pointSize() - 3))
        painter.setFont(status_font)
        painter.setPen(CHAMPAGNE if selected else MUTED)
        status_rect = QRect(rect.left(), rect.top() + 26, rect.width(), 16)
        painter.drawText(
            status_rect,
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
            "Selected" if selected else "Click to add",
        )
        painter.restore()

    def sizeHint(self, option, index) -> QSize:  # noqa: ARG002
        return QSize(130, 62)


class TickerGridList(QListWidget):
    """Two-column wrapping card list with click-to-toggle selection."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("tickerGrid")
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setFlow(QListView.Flow.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setSpacing(4)
        self.setUniformItemSizes(True)
        self.setItemDelegate(TickerCardDelegate(self))
        self.viewport().setCursor(Qt.CursorShape.PointingHandCursor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_grid_size()

    def _update_grid_size(self) -> None:
        viewport_width = max(self.viewport().width(), 200)
        columns = 2
        spacing = self.spacing()
        card_width = max(110, (viewport_width - spacing * (columns + 1)) // columns)
        self.setGridSize(QSize(card_width, 62))


class ConfigurationPanel(QWidget):
    """
    Asset/date/risk-free-rate controls plus a "Run Optimization" trigger.

    Emits `run_requested` with the fully-resolved inputs rather than
    letting the parent window reach into internal widgets directly --
    keeps this panel's internal layout free to change without breaking
    callers.
    """

    run_requested = Signal(list, str, str, float, dict)

    def __init__(
        self, available_tickers: List[str], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._available_tickers = available_tickers
        self._updating_dates = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        tabs = QTabWidget()
        tabs.addTab(self._build_universe_tab(), "Universe")
        tabs.addTab(self._build_parameters_tab(), "Parameters")
        tabs.addTab(self._build_constraints_tab(), "Constraints")
        layout.addWidget(tabs, 1)

        self._run_button = QPushButton("Generate Portfolio")
        self._run_button.setObjectName("generateButton")
        self._run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self._run_button)

    def _build_universe_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        self._count_label = QLabel()
        self._count_label.setObjectName("countLabel")
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_selected)
        header_row.addWidget(self._count_label, 1)
        header_row.addWidget(clear_button)
        layout.addLayout(header_row)

        selected_frame = QFrame()
        selected_frame.setObjectName("selectedSummary")
        selected_layout = QVBoxLayout(selected_frame)
        selected_layout.setContentsMargins(8, 6, 8, 6)
        self._selected_summary = QLabel()
        self._selected_summary.setObjectName("selectedTickers")
        self._selected_summary.setWordWrap(True)
        self._selected_summary.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        selected_layout.addWidget(self._selected_summary)
        selected_frame.setMaximumHeight(72)
        layout.addWidget(selected_frame)

        self._ticker_filter = QLineEdit()
        self._ticker_filter.setPlaceholderText("Filter tickers")
        self._ticker_filter.textChanged.connect(self._filter_tickers)
        layout.addWidget(self._ticker_filter)

        self._ticker_list = TickerGridList()
        default_set = set(DEFAULT_TICKERS)
        self._ticker_list.blockSignals(True)
        for ticker in self._available_tickers:
            item = QListWidgetItem(str(ticker))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._ticker_list.addItem(item)
            item.setSelected(str(ticker) in default_set)
        self._ticker_list.blockSignals(False)
        self._ticker_list.itemSelectionChanged.connect(self._refresh_selected_summary)
        layout.addWidget(self._ticker_list, 1)

        self._refresh_selected_summary()
        return tab

    def _build_parameters_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        today = QDate.currentDate()
        default_start = today.addYears(-DEFAULT_LOOKBACK_YEARS)
        earliest_allowed = today.addYears(-20)

        self._lookback_combo = QComboBox()
        self._lookback_combo.addItems(list(LOOKBACK_PRESETS.keys()))
        self._lookback_combo.setCurrentText("5 Years")
        self._lookback_combo.currentTextChanged.connect(self._on_lookback_changed)
        layout.addRow("Lookback", self._lookback_combo)

        self._start_date_edit = QDateEdit(default_start)
        self._start_date_edit.setCalendarPopup(True)
        self._start_date_edit.setMinimumDate(earliest_allowed)
        self._start_date_edit.setMaximumDate(today)
        self._start_date_edit.dateChanged.connect(self._on_date_edited)

        self._end_date_edit = QDateEdit(today)
        self._end_date_edit.setCalendarPopup(True)
        self._end_date_edit.setMinimumDate(earliest_allowed)
        self._end_date_edit.setMaximumDate(today)
        self._end_date_edit.dateChanged.connect(self._on_date_edited)

        layout.addRow("Start", self._start_date_edit)
        layout.addRow("End", self._end_date_edit)

        slider_row = QWidget()
        slider_layout = QHBoxLayout(slider_row)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        self._risk_free_slider = QSlider(Qt.Orientation.Horizontal)
        self._risk_free_slider.setRange(0, RISK_FREE_SLIDER_MAX_STEPS)
        self._risk_free_slider.setValue(0)
        self._risk_free_value_label = QLabel("0.00%")
        self._risk_free_slider.valueChanged.connect(self._on_risk_free_changed)
        slider_layout.addWidget(self._risk_free_slider)
        slider_layout.addWidget(self._risk_free_value_label)
        layout.addRow("Risk-free rate", slider_row)
        return tab

    def _build_constraints_tab(self) -> QWidget:
        tab = QWidget()
        layout = QFormLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        caption = QLabel(
            "Optional. Leave inactive for unconstrained Max Sharpe. "
            "Enable a control to enforce it."
        )
        caption.setObjectName("hintLabel")
        caption.setWordWrap(True)
        layout.addRow(caption)

        self._min_return_check = QCheckBox("Minimum annualized return")
        self._min_return_spin = QDoubleSpinBox()
        self._min_return_spin.setRange(-50.0, 200.0)
        self._min_return_spin.setDecimals(1)
        self._min_return_spin.setSuffix(" %")
        self._min_return_spin.setValue(10.0)
        self._min_return_spin.setEnabled(False)
        self._min_return_check.toggled.connect(self._min_return_spin.setEnabled)
        layout.addRow(self._min_return_check, self._min_return_spin)

        self._max_volatility_check = QCheckBox("Maximum annualized volatility")
        self._max_volatility_spin = QDoubleSpinBox()
        self._max_volatility_spin.setRange(0.0, 200.0)
        self._max_volatility_spin.setDecimals(1)
        self._max_volatility_spin.setSuffix(" %")
        self._max_volatility_spin.setValue(20.0)
        self._max_volatility_spin.setEnabled(False)
        self._max_volatility_check.toggled.connect(
            self._max_volatility_spin.setEnabled
        )
        layout.addRow(self._max_volatility_check, self._max_volatility_spin)

        # Per-asset allocation band. Applied as the same [min, max] bound
        # to every selected ticker -- e.g. (0%, 25%) caps any single
        # position at a quarter of the portfolio.
        self._min_weight_spin = QDoubleSpinBox()
        self._min_weight_spin.setRange(0.0, 100.0)
        self._min_weight_spin.setDecimals(0)
        self._min_weight_spin.setSuffix(" %")
        self._min_weight_spin.setValue(0.0)

        self._max_weight_spin = QDoubleSpinBox()
        self._max_weight_spin.setRange(0.0, 100.0)
        self._max_weight_spin.setDecimals(0)
        self._max_weight_spin.setSuffix(" %")
        self._max_weight_spin.setValue(100.0)

        layout.addRow("Min allocation / asset", self._min_weight_spin)
        layout.addRow("Max allocation / asset", self._max_weight_spin)
        return tab

    def _filter_tickers(self, text: str) -> None:
        needle = text.strip().lower()
        for index in range(self._ticker_list.count()):
            item = self._ticker_list.item(index)
            ticker = item.text().lower()
            item.setHidden(bool(needle) and needle not in ticker)

    def _clear_selected(self) -> None:
        self._ticker_list.clearSelection()
        self._refresh_selected_summary()

    def _refresh_selected_summary(self) -> None:
        tickers = self._selected_tickers()
        self._count_label.setText(f"{len(tickers)} selected")
        self._selected_summary.setText(", ".join(tickers) if tickers else "None")

    def _on_lookback_changed(self, preset: str) -> None:
        years = LOOKBACK_PRESETS.get(preset)
        if years is None:
            return
        today = QDate.currentDate()
        self._updating_dates = True
        self._end_date_edit.setDate(today)
        self._start_date_edit.setDate(today.addYears(-years))
        self._updating_dates = False

    def _on_date_edited(self) -> None:
        if self._updating_dates:
            return
        self._lookback_combo.blockSignals(True)
        self._lookback_combo.setCurrentText("Custom")
        self._lookback_combo.blockSignals(False)

    def _on_risk_free_changed(self, raw_value: int) -> None:
        rate = raw_value * RISK_FREE_STEP
        self._risk_free_value_label.setText(f"{rate * 100:.2f}%")

    def _selected_tickers(self) -> List[str]:
        return [item.text() for item in self._ticker_list.selectedItems()]

    def _collect_constraints(self) -> Dict[str, Optional[float]]:
        """
        Gather the optional Generate Portfolio constraints as a plain dict.

        Values are converted from the percentage scale shown in the UI
        to the fractional scale (e.g. 0.10) that `PortfolioOptimizer`
        expects.

        Returns:
            Dict[str, Optional[float]]: 'min_return' and 'max_volatility'
                are each a fraction, or `None` if their checkbox is
                unchecked (no constraint). 'min_weight' and 'max_weight'
                are always populated fractions, defaulting to the
                unconstrained `(0.0, 1.0)` allocation range.
        """
        return {
            "min_return": (
                self._min_return_spin.value() / 100.0
                if self._min_return_check.isChecked()
                else None
            ),
            "max_volatility": (
                self._max_volatility_spin.value() / 100.0
                if self._max_volatility_check.isChecked()
                else None
            ),
            "min_weight": self._min_weight_spin.value() / 100.0,
            "max_weight": self._max_weight_spin.value() / 100.0,
        }

    def _on_run_clicked(self) -> None:
        tickers = self._selected_tickers()
        start_date: dt.date = self._start_date_edit.date().toPython()
        end_date: dt.date = self._end_date_edit.date().toPython()
        risk_free_rate = self._risk_free_slider.value() * RISK_FREE_STEP
        constraints = self._collect_constraints()
        self.run_requested.emit(
            tickers,
            start_date.isoformat(),
            end_date.isoformat(),
            risk_free_rate,
            constraints,
        )

    def set_enabled(self, enabled: bool) -> None:
        """Disable the trigger while a background optimization is running."""
        self._run_button.setEnabled(enabled)
        self._run_button.setText("Generate Portfolio" if enabled else "Running...")


# ---------------------------------------------------------------------------
# 5. MetricCard / DonutChart -- small display widgets
# ---------------------------------------------------------------------------


class MetricCard(QFrame):
    """A single labeled statistic, styled to resemble st.metric."""

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        title_label.setAutoFillBackground(False)
        self._value_label = QLabel("--")
        self._value_label.setObjectName("metricValue")
        self._value_label.setAutoFillBackground(False)
        layout.addWidget(title_label)
        layout.addWidget(self._value_label)

    def set_value(self, text: str) -> None:
        self._value_label.setText(text)


class DonutChart(FigureCanvasQTAgg):
    """
    Matplotlib donut chart embedded as a native Qt widget.

    Native rendering avoids pulling in QtWebEngine (and its ~100MB+
    footprint) just to draw one chart, at the cost of Plotly's built-in
    hover/zoom interactivity.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._figure = Figure(figsize=(5, 3.6), facecolor=CHART_SURFACE)
        super().__init__(self._figure)
        self.setParent(parent)
        self._axes = self._figure.add_subplot(111)
        self._axes.set_facecolor(CHART_SURFACE)
        self._axes.axis("off")

    def render(self, weights: Dict[str, float]) -> None:
        self._axes.clear()
        self._axes.set_facecolor(CHART_SURFACE)
        if not weights:
            self._axes.text(
                0.5,
                0.5,
                "No allocation",
                ha="center",
                va="center",
                color="#8c7a66",
            )
            self._axes.axis("off")
        else:
            labels = [
                f"{ticker}  {weight * 100:.1f}%"
                for ticker, weight in weights.items()
            ]
            colors = [
                ALLOCATION_COLORS[index % len(ALLOCATION_COLORS)]
                for index in range(len(weights))
            ]
            wedges, _ = self._axes.pie(
                list(weights.values()),
                colors=colors,
                wedgeprops=dict(width=0.42, edgecolor=CHART_SURFACE),
                startangle=90,
            )
            self._axes.legend(
                wedges,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=8,
            )
            self._axes.set_aspect("equal")
        self._figure.subplots_adjust(left=0.04, right=0.62, top=0.95, bottom=0.05)
        self.draw()


# ---------------------------------------------------------------------------
# 6. ResultsPanel -- metrics + chart + allocation table
# ---------------------------------------------------------------------------


class ResultsPanel(QWidget):
    """Headline metrics + donut chart + allocation table."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("resultsPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_empty_state())
        self._stack.addWidget(self._build_results_view())
        layout.addWidget(self._stack, 1)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

    def _build_empty_state(self) -> QWidget:
        empty = QWidget()
        layout = QVBoxLayout(empty)
        layout.addStretch(1)
        title = QLabel("No portfolio generated")
        title.setObjectName("emptyStateTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        body = QLabel(
            "Select a universe, set the lookback and risk-free rate, "
            "then generate a portfolio to view allocation and risk metrics."
        )
        body.setObjectName("emptyStateBody")
        body.setWordWrap(True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addStretch(1)
        return empty

    def _build_results_view(self) -> QWidget:
        results = QWidget()
        layout = QVBoxLayout(results)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)
        self._return_card = MetricCard("Annualized Return")
        self._volatility_card = MetricCard("Annualized Volatility")
        self._sharpe_card = MetricCard("Sharpe Ratio")
        for card in (self._return_card, self._volatility_card, self._sharpe_card):
            metrics_row.addWidget(card)
        layout.addLayout(metrics_row)

        self._results_tabs = QTabWidget()
        self._donut_chart = DonutChart()
        self._results_tabs.addTab(self._donut_chart, "Allocation")

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.setContentsMargins(8, 8, 8, 8)
        self._allocation_table = QTableWidget(0, 2)
        self._allocation_table.setHorizontalHeaderLabels(["Ticker", "Allocation"])
        self._allocation_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self._allocation_table.verticalHeader().setVisible(False)
        self._allocation_table.setAlternatingRowColors(True)
        self._allocation_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._allocation_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        table_layout.addWidget(self._allocation_table)
        self._results_tabs.addTab(table_page, "Holdings")
        layout.addWidget(self._results_tabs, 1)
        return results

    def show_result(self, result: dict) -> None:
        self._set_status("", "info")
        self._return_card.set_value(f"{result['return'] * 100:.2f}%")
        self._volatility_card.set_value(f"{result['volatility'] * 100:.2f}%")
        self._sharpe_card.set_value(f"{result['sharpe_ratio']:.2f}")

        weights: Dict[str, float] = result["weights"]
        non_zero_weights = {
            ticker: weight
            for ticker, weight in weights.items()
            if weight > ZERO_WEIGHT_TOLERANCE
        }
        self._donut_chart.render(non_zero_weights)
        self._populate_table(weights)
        self._stack.setCurrentIndex(1)
        self._results_tabs.setCurrentIndex(0)

    def _populate_table(self, weights: Dict[str, float]) -> None:
        allocation_df = self._build_allocation_dataframe(weights)
        self._allocation_table.setRowCount(len(allocation_df))
        for row, (_, record) in enumerate(allocation_df.iterrows()):
            self._allocation_table.setItem(row, 0, QTableWidgetItem(record["Ticker"]))
            self._allocation_table.setItem(
                row, 1, QTableWidgetItem(record["Allocation"])
            )

    @staticmethod
    def _build_allocation_dataframe(weights: Dict[str, float]) -> pd.DataFrame:
        allocation_df = (
            pd.Series(weights, name="Weight").rename_axis("Ticker").reset_index()
        )
        allocation_df = allocation_df[allocation_df["Weight"] > ZERO_WEIGHT_TOLERANCE]
        allocation_df = allocation_df.sort_values("Weight", ascending=False)
        allocation_df["Allocation"] = allocation_df["Weight"].apply(
            lambda w: f"{w * 100:.2f}%"
        )
        return allocation_df[["Ticker", "Allocation"]].reset_index(drop=True)

    def show_busy(self, message: str) -> None:
        self._set_status(message, "info")

    def show_warning(self, message: str) -> None:
        self._set_status(message, "warning")

    def show_error(self, message: str) -> None:
        self._set_status(message, "error")

    def _set_status(self, message: str, kind: str) -> None:
        object_names = {
            "warning": "statusWarning",
            "error": "statusError",
            "info": "statusInfo",
        }
        self._status_label.setObjectName(object_names.get(kind, "statusInfo"))
        self._status_label.setText(message)
        self._status_label.style().unpolish(self._status_label)
        self._status_label.style().polish(self._status_label)


# ---------------------------------------------------------------------------
# 7. MainWindow -- wires everything together
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Quantitative Portfolio Optimization Dashboard main window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quantitative Portfolio Optimization Dashboard")
        self.resize(1180, 760)

        self._data_cache = DataCache(DATA_FILE_PATH)
        self._worker: Optional[OptimizationWorker] = None

        available_tickers = self._load_available_tickers()

        self._config_panel = ConfigurationPanel(available_tickers)
        self._config_panel.setMinimumWidth(320)
        self._config_panel.setMaximumWidth(400)
        self._results_panel = ResultsPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._config_panel)
        splitter.addWidget(self._results_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 840])
        splitter.setHandleWidth(1)

        root = QWidget()
        root.setObjectName("windowRoot")
        root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_header())
        root_layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

        self._config_panel.run_requested.connect(self._on_run_requested)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("headerBar")
        header.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(header)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(4)
        title = QLabel("Portfolio Optimization")
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle = QLabel("Max Sharpe allocation from historical returns")
        subtitle.setObjectName("appSubtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return header

    def _load_available_tickers(self) -> List[str]:
        try:
            return self._data_cache.get_available_tickers()
        except ValueError as error:
            QMessageBox.critical(self, "Dataset Error", str(error))
            return []

    def _on_run_requested(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        risk_free_rate: float,
        constraints: Dict[str, Optional[float]],
    ) -> None:
        if len(tickers) < 2:
            self._results_panel.show_warning(
                "Please select at least 2 assets to optimize a portfolio."
            )
            return

        # Guard against overlapping runs -- ignore a new request while one
        # is already in flight rather than racing two threads against the
        # same ResultsPanel widgets.
        if self._worker is not None and self._worker.isRunning():
            return

        self._config_panel.set_enabled(False)
        self._results_panel.show_busy("Running optimization…")
        self._worker = OptimizationWorker(
            self._data_cache,
            tickers,
            start_date,
            end_date,
            risk_free_rate,
            constraints,
        )
        self._worker.succeeded.connect(self._results_panel.show_result)
        self._worker.failed.connect(self._results_panel.show_error)
        self._worker.finished.connect(lambda: self._config_panel.set_enabled(True))
        self._worker.start()


# ---------------------------------------------------------------------------
# 8. Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Quantitative Portfolio Optimization Dashboard")
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setFont(QFont("Helvetica Neue", 13))
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
