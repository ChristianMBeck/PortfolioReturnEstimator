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
    4. ConfigurationPanel -- left sidebar: assets, dates, risk-free rate
    5. MetricCard / DonutChart -- small display widgets
    6. ResultsPanel       -- metrics + chart + allocation table
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
from PySide6.QtCore import QDate, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDateEdit,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
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
 
PriceCacheKey = Tuple[str, Tuple[str, ...], str, str]
 
 
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
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._data_cache = data_cache
        self._tickers = tickers
        self._start_date = start_date
        self._end_date = end_date
        self._risk_free_rate = risk_free_rate
 
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
 
 
class ConfigurationPanel(QWidget):
    """
    Asset/date/risk-free-rate controls plus a "Run Optimization" trigger.
 
    Emits `run_requested` with the fully-resolved inputs rather than
    letting the parent window reach into internal widgets directly --
    keeps this panel's internal layout free to change without breaking
    callers.
    """
 
    run_requested = Signal(list, str, str, float)
 
    def __init__(
        self, available_tickers: List[str], parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self._available_tickers = available_tickers
        self._build_ui()
 
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
 
        # -- Asset selection --------------------------------------------
        asset_group = QGroupBox("Select Assets")
        asset_layout = QVBoxLayout(asset_group)
        self._ticker_list = QListWidget()
        self._ticker_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )
        for ticker in self._available_tickers:
            item = QListWidgetItem(ticker)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if ticker in DEFAULT_TICKERS
                else Qt.CheckState.Unchecked
            )
            self._ticker_list.addItem(item)
        asset_layout.addWidget(self._ticker_list)
        layout.addWidget(asset_group)
 
        # -- Date range -----------------------------------------------------
        date_group = QGroupBox("Optimization Timeframe")
        date_layout = QFormLayout(date_group)
 
        today = QDate.currentDate()
        default_start = today.addYears(-DEFAULT_LOOKBACK_YEARS)
        earliest_allowed = today.addYears(-20)
 
        self._start_date_edit = QDateEdit(default_start)
        self._start_date_edit.setCalendarPopup(True)
        self._start_date_edit.setMinimumDate(earliest_allowed)
        self._start_date_edit.setMaximumDate(today)
 
        self._end_date_edit = QDateEdit(today)
        self._end_date_edit.setCalendarPopup(True)
        self._end_date_edit.setMinimumDate(earliest_allowed)
        self._end_date_edit.setMaximumDate(today)
 
        date_layout.addRow("Start", self._start_date_edit)
        date_layout.addRow("End", self._end_date_edit)
        layout.addWidget(date_group)
 
        # -- Risk-free rate ---------------------------------------------------
        risk_group = QGroupBox("Risk-Free Rate")
        risk_layout = QVBoxLayout(risk_group)
        slider_row = QHBoxLayout()
        self._risk_free_slider = QSlider(Qt.Orientation.Horizontal)
        self._risk_free_slider.setRange(0, RISK_FREE_SLIDER_MAX_STEPS)
        self._risk_free_slider.setValue(0)
        self._risk_free_value_label = QLabel("0.0000")
        self._risk_free_slider.valueChanged.connect(self._on_risk_free_changed)
        slider_row.addWidget(self._risk_free_slider)
        slider_row.addWidget(self._risk_free_value_label)
        risk_layout.addLayout(slider_row)
        layout.addWidget(risk_group)
 
        # -- Run button -----------------------------------------------------
        self._run_button = QPushButton("Run Optimization")
        self._run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self._run_button)
        layout.addStretch(1)
 
    def _on_risk_free_changed(self, raw_value: int) -> None:
        rate = raw_value * RISK_FREE_STEP
        self._risk_free_value_label.setText(f"{rate:.4f}")
 
    def _selected_tickers(self) -> List[str]:
        selected = []
        for index in range(self._ticker_list.count()):
            item = self._ticker_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(item.text())
        return selected
 
    def _on_run_clicked(self) -> None:
        tickers = self._selected_tickers()
        start_date: dt.date = self._start_date_edit.date().toPython()
        end_date: dt.date = self._end_date_edit.date().toPython()
        risk_free_rate = self._risk_free_slider.value() * RISK_FREE_STEP
        self.run_requested.emit(
            tickers, start_date.isoformat(), end_date.isoformat(), risk_free_rate
        )
 
    def set_enabled(self, enabled: bool) -> None:
        """Disable the trigger while a background optimization is running."""
        self._run_button.setEnabled(enabled)
        self._run_button.setText("Run Optimization" if enabled else "Running...")
 
 
# ---------------------------------------------------------------------------
# 5. MetricCard / DonutChart -- small display widgets
# ---------------------------------------------------------------------------
 
 
class MetricCard(QFrame):
    """A single labeled statistic, styled to resemble st.metric."""
 
    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #666; font-size: 12px;")
        self._value_label = QLabel("--")
        self._value_label.setStyleSheet("font-size: 24px; font-weight: 600;")
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
        self._figure = Figure(figsize=(4, 4))
        super().__init__(self._figure)
        self.setParent(parent)
        self._axes = self._figure.add_subplot(111)
 
    def render(self, weights: Dict[str, float]) -> None:
        self._axes.clear()
        if not weights:
            self._axes.text(0.5, 0.5, "No allocation", ha="center", va="center")
        else:
            wedges, _ = self._axes.pie(
                list(weights.values()), wedgeprops=dict(width=0.4), startangle=90
            )
            self._axes.legend(
                wedges,
                [f"{ticker} ({weight * 100:.1f}%)" for ticker, weight in weights.items()],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=8,
            )
        self._axes.set_aspect("equal")
        self._figure.tight_layout()
        self.draw()
 
 
# ---------------------------------------------------------------------------
# 6. ResultsPanel -- metrics + chart + allocation table
# ---------------------------------------------------------------------------
 
 
class ResultsPanel(QWidget):
    """Headline metrics + donut chart + allocation table."""
 
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()
 
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
 
        # -- Metrics row --------------------------------------------------
        metrics_row = QHBoxLayout()
        self._return_card = MetricCard("Annualized Return")
        self._volatility_card = MetricCard("Annualized Volatility")
        self._sharpe_card = MetricCard("Sharpe Ratio")
        for card in (self._return_card, self._volatility_card, self._sharpe_card):
            metrics_row.addWidget(card)
        layout.addLayout(metrics_row)
 
        # -- Chart + table --------------------------------------------------
        results_grid = QGridLayout()
 
        chart_group = QGroupBox("Optimal Allocation")
        chart_layout = QVBoxLayout(chart_group)
        self._donut_chart = DonutChart()
        chart_layout.addWidget(self._donut_chart)
        results_grid.addWidget(chart_group, 0, 0)
 
        table_group = QGroupBox("Allocation Breakdown")
        table_layout = QVBoxLayout(table_group)
        self._allocation_table = QTableWidget(0, 2)
        self._allocation_table.setHorizontalHeaderLabels(["Ticker", "Allocation"])
        self._allocation_table.horizontalHeader().setStretchLastSection(True)
        self._allocation_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        table_layout.addWidget(self._allocation_table)
        results_grid.addWidget(table_group, 0, 1)
 
        layout.addLayout(results_grid)
 
        self._status_label = QLabel("")
        layout.addWidget(self._status_label)
 
    def show_result(self, result: dict) -> None:
        self._status_label.setText("")
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
 
    def show_warning(self, message: str) -> None:
        self._status_label.setStyleSheet("color: #8a6d3b;")
        self._status_label.setText(message)
 
    def show_error(self, message: str) -> None:
        self._status_label.setStyleSheet("color: #b00020;")
        self._status_label.setText(message)
 
 
# ---------------------------------------------------------------------------
# 7. MainWindow -- wires everything together
# ---------------------------------------------------------------------------
 
 
class MainWindow(QMainWindow):
    """Quantitative Portfolio Optimization Dashboard main window."""
 
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Quantitative Portfolio Optimization Dashboard")
        self.resize(1200, 800)
 
        self._data_cache = DataCache(DATA_FILE_PATH)
        self._worker: Optional[OptimizationWorker] = None
 
        available_tickers = self._load_available_tickers()
 
        self._config_panel = ConfigurationPanel(available_tickers)
        self._results_panel = ResultsPanel()
 
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._config_panel)
        splitter.addWidget(self._results_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 880])
        self.setCentralWidget(splitter)
 
        self._config_panel.run_requested.connect(self._on_run_requested)
 
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
        self._worker = OptimizationWorker(
            self._data_cache, tickers, start_date, end_date, risk_free_rate
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
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
 
 
if __name__ == "__main__":
    main()
 


