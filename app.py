"""
Hispanic Heritage Historical Data Explorer
A Streamlit teaching app for World Bank historical data and interactive function analysis.

Run: python -m streamlit run app.py
"""
from __future__ import annotations

import io
import math
import re
import time
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PDFImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

APP_TITLE = "Hispanic Heritage Historical Data Explorer"
COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Average number of years a newborn would live if current age-specific mortality rates remained constant.", "Higher values indicate longer life expectancy; this is a population indicator, not a guarantee for an individual."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates.", "Definitions and comparability can vary by country and over time."),
    "GDP per capita (current US$)": ("NY.GDP.PCAP.CD", "current US$ per person", "Gross domestic product divided by midyear population, in current US dollars.", "Not average wealth or household income; current-dollar values are affected by prices and exchange rates."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Live births during a year per 1,000 people in the midyear population.", "A rate, not a count of births."),
    "Net migration": ("SM.POP.NETM", "people (net)", "Net total of immigrants and emigrants over a period; positive values indicate net inflow.", "Not an emigration count; estimates can be irregular and revised."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "deaths per 100,000 people", "Victims of intentional homicide per 100,000 population.", "Definitions and reporting quality may differ across countries and years."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index.", "Inflation rate is not the price level itself."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP.", "Coverage is central government and may not equal general-government debt."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, expressed in current US dollars.", "Nominal dollar amounts are affected by exchange rates and prices."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services expressed as a percentage of GDP.", "This is not a full current-account balance."),
}
WB_ROOT = "https://api.worldbank.org/v2"
THIS_YEAR = date.today().year

CONTEXTS = {
    "Business": ("Revenue", "thousand dollars", "Time", "years", "A revenue-like output over time", "polynomial"),
    "Engineering": ("Position", "meters", "Time", "seconds", "Position of a moving object", "polynomial"),
    "Medical": ("Concentration", "mg/L", "Time", "hours", "Illustrative concentration over time; not medical advice", "exponential"),
    "Sports science": ("Position", "meters", "Time", "seconds", "Movement during a training drill", "trigonometric"),
    "Biology": ("Population", "organisms", "Time", "days", "Population or cell-colony size", "exponential"),
    "Bioengineering": ("Signal", "arbitrary units", "Time", "seconds", "An illustrative sensor or biological signal", "trigonometric"),
    "Economic forecasting": ("Indicator", "index points", "Time", "years", "An economic indicator trend", "polynomial"),
    "Weather": ("Temperature", "°C", "Time", "hours", "An illustrative temperature cycle", "trigonometric"),
    "Custom": ("Output", "units", "Input", "units", "Describe your own model and assumptions", "polynomial"),
}
FUNCTION_PRESETS = {
    "Polynomial": "a*x**2 + b*x + c",
    "Rational": "(a*x + b)/(x - h)",
    "Exponential": "a*exp(b*x) + c",
    "Logarithmic": "a*log(x - h) + c",
    "Trigonometric": "a*sin(b*x) + c",
}
PARAM_DEFAULTS = {"a": 1.0, "b": 0.2, "c": 0.0, "h": -1.0}
SYMBOLS = {"x": sp.Symbol("x", real=True), "a": sp.Symbol("a", real=True),
           "b": sp.Symbol("b", real=True), "c": sp.Symbol("c", real=True),
           "h": sp.Symbol("h", real=True)}
ALLOWED_FUNCS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
                 "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs}


def world_bank_url(country: str, indicator: str, start: int, end: int) -> str:
    return f"{WB_ROOT}/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=20000"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_indicator(country: str, indicator: str, start: int, end: int) -> tuple[pd.DataFrame, str]:
    """Fetch one official World Bank indicator. Returns (year/value dataframe, status)."""
    url = world_bank_url(country, indicator, start, end)
    try:
        response = requests.get(url, timeout=(5, 20), headers={"User-Agent": "HispanicHeritageDataExplorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            return pd.DataFrame(columns=["year", "value"]), "The API returned an empty or unexpected response."
        rows = []
        for item in payload[1]:
            if item and item.get("date") and item.get("value") is not None:
                try:
                    rows.append({"year": int(item["date"]), "value": float(item["value"])})
                except (TypeError, ValueError):
                    continue
        df = pd.DataFrame(rows, columns=["year", "value"]).drop_duplicates("year").sort_values("year")
        if df.empty:
            return df, "The API returned no non-null observations for this selection."
        return df.reset_index(drop=True), f"Retrieved {len(df)} non-null observations."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Retry later or check network access."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"World Bank request failed: {exc}"
    except (ValueError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not decode the API response: {exc}"


def validate_working_data(raw: pd.DataFrame, indicator_name: str) -> tuple[pd.DataFrame, list[str]]:
    errors = []
    if not {"year", "value"}.issubset(raw.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    df = raw[["year", "value"]].copy()
    years = pd.to_numeric(df["year"], errors="coerce")
    values = pd.to_numeric(df["value"], errors="coerce")
    if years.isna().any() or ((years % 1) != 0).any():
        errors.append("Every year must be a whole integer.")
    if years.dropna().duplicated().any():
        errors.append("Duplicate years are not allowed.")
    if values.isna().any() or not np.isfinite(values.dropna()).all():
        errors.append("Values must be finite numbers; blanks and infinity are not allowed.")
    df["year"], df["value"] = years, values
    df = df.dropna(subset=["year", "value"])
    if indicator_name == "Life expectancy at birth" and ((df.value < 0) | (df.value > 125)).any():
        errors.append("Life expectancy should be between 0 and 125 years.")
    if indicator_name == "Unemployment, national estimate" and ((df.value < 0) | (df.value > 100)).any():
        errors.append("Unemployment percentage should be between 0 and 100.")
    if indicator_name == "Crude birth rate" and (df.value < 0).any():
        errors.append("Crude birth rate cannot be negative.")
    if indicator_name == "Intentional homicide rate" and (df.value < 0).any():
        errors.append("Homicide rate cannot be negative.")
    if indicator_name == "Central government debt" and (df.value < 0).any():
        errors.append("Central government debt percentage cannot be negative.")
    return df.sort_values("year").reset_index(drop=True), errors


def fit_polynomial(df: pd.DataFrame, degree: int):
    clean, errors = validate_working_data(df, st.session_state.get("indicator_name", ""))
    if errors:
        raise ValueError(" ".join(errors))
    clean = clean.sort_values("year")
    if clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years; only {clean.year.nunique()} are available.")
    base = int(clean.year.min())
    t = clean.year.to_numpy(dtype=float) - base
    y = clean.value.to_numpy(dtype=float)
    coef = np.polynomial.polynomial.polyfit(t, y, degree)
    pred = np.polynomial.polynomial.polyval(t, coef)
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    return {"base": base, "degree": degree, "coef": coef, "years": clean.year.to_numpy(),
            "values": y, "fitted": pred, "r2": r2, "mae": mae, "n": len(y),
            "min_year": int(clean.year.min()), "max_year": int(clean.year.max()),
            "mean": float(np.mean(y)), "residuals": y - pred, "data": clean}


def polynomial_equation(model: dict, unit: str = "") -> str:
    terms = []
    for power, coefficient in enumerate(model["coef"]):
        if abs(coefficient) < 1e-12:
            continue
        if power == 0:
            terms.append(f"{coefficient:.10g}")
        elif power == 1:
            terms.append(f"{coefficient:+.10g}t")
        else:
            terms.append(f"{coefficient:+.10g}t^{power}")
    return "y = " + (" ".join(terms) if terms else "0") + f", where t = year − {model['base']}"


def predict_poly(model: dict, year: float) -> float:
    return float(np.polynomial.polynomial.polyval(float(year) - model["base"], model["coef"]))


def safe_parse_function(text: str, params: dict[str, float]):
    """Parse a student expression using a small SymPy whitelist; x is the input variable."""
    expression = text.strip().replace("^", "**")
    if not expression or len(expression) > 250:
        raise ValueError("Enter a function of x (1–250 characters).")
    # Reject attribute access, indexing, imports, and characters outside mathematical notation.
    if not re.fullmatch(r"[A-Za-z0-9_+\-*/().,\s**]+", expression):
        raise ValueError("Use only numbers, x, a, b, c, h, parentheses, and standard operators.")
    local = dict(SYMBOLS)
    local.update(ALLOWED_FUNCS)
    try:
        expr = sp.sympify(expression, locals=local, evaluate=True)
        substitutions = {SYMBOLS[k]: float(v) for k, v in params.items() if k in SYMBOLS and k != "x"}
        expr = sp.simplify(expr.subs(substitutions))
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("The expression contains an undefined or infinite constant.")
        if any(s not in (SYMBOLS["x"],) for s in expr.free_symbols):
            raise ValueError("Unrecognized variable. Use x as the independent variable.")
        return expr
    except (sp.SympifyError, TypeError, SyntaxError) as exc:
        raise ValueError("Could not parse that function. Try notation such as a*x**2+b*x+c or sin(x).") from exc


def lambdify_real(expr):
    fn = sp.lambdify(SYMBOLS["x"], expr, modules=["numpy"])
    def wrapped(values):
        arr = np.asarray(values, dtype=float)
        with np.errstate(all="ignore"):
            out = fn(arr)
        if np.isscalar(out):
            out = np.full(arr.shape, out, dtype=float)
        out = np.asarray(out, dtype=float)
        if out.shape == ():
            out = np.full(arr.shape, float(out))
        return np.where(np.isfinite(out), out, np.nan)
    return wrapped


def real_value(fn, x: float) -> float:
    try:
        value = np.asarray(fn(np.array([float(x)]))).reshape(-1)[0]
        return float(value) if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def numeric_roots(expr, lo, hi, samples=2500):
    fn = lambdify_real(expr)
    grid = np.linspace(lo, hi, samples)
    vals = fn(grid)
    roots = []
    for i in range(len(grid) - 1):
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        if vals[i] == 0:
            roots.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0:
            # Bisection is robust and avoids requiring SciPy.
            left, right = grid[i], grid[i + 1]
            fl = vals[i]
            for _ in range(45):
                mid = (left + right) / 2
                fm = real_value(fn, mid)
                if not np.isfinite(fm):
                    break
                if fl * fm <= 0:
                    right = mid
                else:
                    left, fl = mid, fm
            roots.append(float((left + right) / 2))
    return dedupe_sorted(roots, tol=(hi-lo)/samples*2)


def dedupe_sorted(vals, tol=1e-3):
    out = []
    for val in sorted(vals):
        if not out or abs(val - out[-1]) > tol:
            out.append(val)
    return out


def make_plot(x, y, title, xlabel, ylabel, points=None, point_labels=None, curves=None,
              vlines=None, sample_step=1, show_grid=True):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    if curves:
        for xx, yy, label, style in curves:
            ax.plot(xx, yy, style, linewidth=2 if "fit" in label.lower() else 1.8, label=label)
    if points is not None:
        ax.scatter(points[0], points[1], label=point_labels or "Observed data", s=32, zorder=4)
    if vlines:
        for xv, label in vlines:
            ax.axvline(xv, linestyle="--", linewidth=1.2, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_grid:
        ax.grid(True, alpha=.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def make_pdf(metadata: dict, chart_png: bytes | None, preview: pd.DataFrame, results: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=0.45*inch,
                            leftMargin=0.45*inch, topMargin=0.4*inch, bottomMargin=0.4*inch)
    styles = getSampleStyleSheet()
    story = [Paragraph(APP_TITLE, styles["Title"]), Spacer(1, 6)]
    for line in metadata.get("lines", []):
        story.append(Paragraph(str(line).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles["BodyText"]))
    story.append(Spacer(1, 8))
    if chart_png:
        story.append(PDFImage(io.BytesIO(chart_png), width=8.8*inch, height=4.3*inch))
        story.append(Spacer(1, 6))
    story.append(Paragraph("Data preview (first 15 rows)", styles["Heading2"]))
    data = [list(preview.columns)] + preview.head(15).astype(str).values.tolist()
    table = Table(data, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0),colors.HexColor("#e7edf4")),
                               ("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),
                               ("VALIGN",(0,0),(-1,-1),"TOP")]))
    story.append(table)
    if not results.empty:
        story += [Spacer(1, 10), Paragraph("Calculated-year results", styles["Heading2"])]
        result_data = [list(results.columns)] + results.astype(str).values.tolist()
        rt = Table(result_data, repeatRows=1, hAlign="LEFT")
        rt.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e7edf4")),
                                ("GRID",(0,0),(-1,-1),0.3,colors.grey),("FONTSIZE",(0,0),(-1,-1),7)]))
        story.append(rt)
    doc.build(story)
    return buffer.getvalue()


st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
st.title("📈 " + APP_TITLE)
st.caption("HISPANIC HERITAGE DATA LAB · historical evidence, mathematical models, and context-aware function analysis")
st.info("Research question starter: How has **[choose an indicator]** changed over time in **[choose countries]**, and what can a mathematical model responsibly tell us? Brazil is included as a Latin American comparison; Latin America is diverse, and Brazil is not a proxy for every country or for Hispanic identity.")

with st.sidebar:
    st.header("Historical data")
    country_label = st.selectbox("Featured country", list(COUNTRIES))
    indicator_name = st.selectbox("World Bank indicator", list(INDICATORS))
    comparison_labels = st.multiselect("Comparison countries", [c for c in COUNTRIES if c != country_label], default=["Brazil (BR)"] if country_label != "Brazil (BR)" else ["Mexico (MX)"])
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 3)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    printer_mode = st.checkbox("Printer-friendly report view", value=False)
    st.divider()
    st.header("Mathematics pathway")
    pathway = st.radio("Course level", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)
    st.caption("The pathway changes which mathematical interpretations and analyses are emphasized.")

indicator_code, indicator_units, indicator_definition, indicator_caveat = INDICATORS[indicator_name]
country_code = COUNTRIES[country_label]
start_year, end_year = THIS_YEAR - 70, THIS_YEAR - 1
retrieved_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
st.markdown(f"**Research question:** How has {indicator_name.lower()} changed in {country_label} from {start_year} to {end_year}, and how well does a polynomial describe the observed pattern?")
st.markdown(f"**Definition:** {indicator_definition}  \n**Units:** {indicator_units}  \n**Official source:** [World Bank indicator page](https://data.worldbank.org/indicator/{indicator_code}) · [API endpoint for this selection]({world_bank_url(country_code, indicator_code, start_year, end_year)})  \n**Requested period:** {start_year}–{end_year} inclusive · **Retrieved:** {retrieved_at}")
st.caption(indicator_caveat)

raw_df, fetch_status = fetch_indicator(country_code, indicator_code, start_year, end_year)
st.write(f"**API status:** {fetch_status}")
if raw_df.empty:
    st.error("No source observations are available for this selection. The model is disabled until data are available; no synthetic values are substituted.")
    st.stop()

source_key = f"source::{country_code}::{indicator_code}::{start_year}::{end_year}"
working_key = f"working::{country_code}::{indicator_code}::{start_year}::{end_year}"
if source_key not in st.session_state:
    st.session_state[source_key] = raw_df.copy()
if working_key not in st.session_state:
    st.session_state[working_key] = raw_df.copy()
if st.session_state.get("last_working_key") != working_key:
    st.session_state["last_working_key"] = working_key
    st.session_state["working_editor_seed"] = st.session_state[working_key].copy()

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Source data and editable working table")
    st.write(f"Actual source coverage: **{raw_df.year.min()}–{raw_df.year.max()}**, {len(raw_df)} non-null observations (requested window contains 70 years, but coverage may be smaller).")
    st.dataframe(raw_df, use_container_width=True, hide_index=True)
with right:
    st.markdown("**Edit the working table**")
    st.caption("Edit years or values for a scenario. Source data above remains unchanged.")
    edited_table = st.data_editor(st.session_state[working_key].copy(), num_rows="dynamic", use_container_width=True, key=f"editor::{working_key}", column_config={
        "year": st.column_config.NumberColumn("Year", step=1, format="%d"),
        "value": st.column_config.NumberColumn(f"Value ({indicator_units})", format="%.6g")
    })
    b1, b2 = st.columns(2)
    if b1.button("Apply and validate edits", use_container_width=True):
        cleaned, errors = validate_working_data(edited_table, indicator_name)
        if errors:
            st.session_state["edit_errors"] = errors
        elif cleaned.empty:
            st.session_state["edit_errors"] = ["The working table must contain at least one valid row."]
        else:
            st.session_state[working_key] = cleaned
            st.session_state["edit_errors"] = []
            st.session_state["working_editor_seed"] = cleaned.copy()
            st.session_state.pop("model_cache_key", None)
            st.success("Validated working scenario applied.")
            st.rerun()
    if b2.button("Reset to source", use_container_width=True):
        st.session_state[working_key] = st.session_state[source_key].copy()
        st.session_state["edit_errors"] = []
        st.session_state.pop("model_cache_key", None)
        st.rerun()
    for err in st.session_state.get("edit_errors", []):
        st.error(err)

working_df, validation_errors = validate_working_data(st.session_state[working_key], indicator_name)
if validation_errors:
    for err in validation_errors:
        st.error(err)
    st.stop()
is_modified = not working_df.equals(st.session_state[source_key].sort_values("year").reset_index(drop=True))
st.caption("Data status: **USER-MODIFIED SCENARIO**" if is_modified else "Data status: source data (unmodified)")
if len(working_df) < degree + 1:
    st.error(f"Not enough distinct valid years for degree {degree}. Need at least {degree + 1}; currently {working_df.year.nunique()}. Lower the degree or restore/add valid observations.")
    st.stop()
try:
    model = fit_polynomial(working_df, degree)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# Cache calculated results by underlying selection and working data fingerprint.
fingerprint = (country_code, indicator_code, degree, tuple(working_df.year), tuple(np.round(working_df.value, 12)))
if st.session_state.get("model_cache_key") != fingerprint:
    st.session_state["model_cache_key"] = fingerprint
    st.session_state["calculated_results"] = pd.DataFrame()
    st.session_state["single_result"] = None

st.header("1. Polynomial fit to historical data")
st.markdown("Time is centered as **t = year − base year** to improve numerical conditioning. Metrics below describe the fitted data, not future certainty.")
equation = polynomial_equation(model)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Observations", model["n"])
m2.metric("R² (in-sample)", f"{model['r2']:.5f}")
m3.metric("MAE (in-sample)", f"{model['mae']:.5g} {indicator_units}")
m4.metric("Historical mean", f"{model['mean']:.5g}")
m5.metric("Fitted range", f"{model['min_year']}–{model['max_year']}")
st.code(equation, language="text")
st.write(f"Base year: **{model['base']}** · Degree: **{degree}** · Residuals are actual minus fitted values. MAE is the average absolute in-sample error.")
with st.expander("Residuals and chronological holdout check"):
    residual_df = pd.DataFrame({"year": model["years"], "actual": model["values"], "fitted": model["fitted"], "residual (actual − fitted)": model["residuals"]})
    st.dataframe(residual_df, use_container_width=True, hide_index=True)
    holdout_n = max(1, int(round(model["n"] * .2)))
    if model["n"] >= max(degree + 3, 8) and model["n"] - holdout_n >= degree + 1:
        train = model["data"].iloc[:-holdout_n]
        test = model["data"].iloc[-holdout_n:]
        t_train = train.year.to_numpy(dtype=float) - model["base"]
        coefs = np.polynomial.polynomial.polyfit(t_train, train.value.to_numpy(dtype=float), degree)
        hold_preds = np.polynomial.polynomial.polyval(test.year.to_numpy(dtype=float) - model["base"], coefs)
        st.write(f"Chronological holdout MAE (last {holdout_n} observations, fit on earlier years): **{mean_absolute_error(test.value, hold_preds):.6g} {indicator_units}**.")
        st.caption("This is a separate out-of-sample-style check for this split; it is not a confidence or prediction interval.")
    else:
        st.caption("Chronological holdout check unavailable: too few observations remain to fit the selected degree and reserve test years.")

fit_min, fit_max = model["min_year"], model["max_year"]
display_max = fit_max + horizon
plot_years = np.linspace(fit_min, display_max, max(500, (display_max-fit_min+1)*5))
plot_vals = np.array([predict_poly(model, y) for y in plot_years])
sample_years = np.arange(fit_min, display_max + 1, sample_step)
sample_vals = np.array([predict_poly(model, y) for y in sample_years])
fig = make_plot(plot_years, plot_vals, f"{indicator_name}: observed data and degree-{degree} fit",
                "Year", f"{indicator_name} ({indicator_units})",
                points=(working_df.year.to_numpy(), working_df.value.to_numpy()), point_labels="Observed / working data",
                curves=[(plot_years[plot_years <= fit_max], plot_vals[plot_years <= fit_max], "Fitted curve (historical range)", "-"),
                        (plot_years[plot_years > fit_max], plot_vals[plot_years > fit_max], "Polynomial extrapolation", "--"),
                        (sample_years, sample_vals, f"Sample markers every {sample_step} years", "o")],
                vlines=[(fit_max, "Historical boundary")])
st.pyplot(fig, use_container_width=True)
plt.close(fig)
st.caption("The smooth fitted curve is distinct from sample markers. Dashed values beyond the observed range are extrapolations, not validated forecasts.")
if horizon:
    ext = np.array([predict_poly(model, fit_max + horizon)])
    if not np.isfinite(ext[0]) or abs(ext[0]) > max(1e6, abs(model["mean"]) * 1000):
        st.warning("The polynomial produces an extreme or non-finite value at the display horizon. This is an extrapolation warning; values are not silently clipped.")
    elif indicator_name in ("Life expectancy at birth", "Unemployment, national estimate", "Crude birth rate", "Intentional homicide rate") and (ext[0] < 0 or (indicator_name == "Life expectancy at birth" and ext[0] > 125) or (indicator_name == "Unemployment, national estimate" and ext[0] > 100)):
        st.warning("The extrapolated endpoint falls outside a basic plausible range for this indicator. The model is not automatically clipped.")
else:
    st.caption("Forward display horizon is 0 years; only the historical range is displayed.")

st.subheader("Single-year estimate")
single_year = st.number_input("Year to calculate (1950–2100)", min_value=1950, max_value=2100, value=min(max(fit_max, 1950), 2100), step=1, key="single_year_input")
if st.button("Calculate Value", type="primary"):
    value = predict_poly(model, int(single_year))
    st.session_state["single_result"] = {"year": int(single_year), "estimate": value, "fingerprint": fingerprint}
single_result = st.session_state.get("single_result")
if single_result and single_result.get("fingerprint") == fingerprint:
    yr, val = single_result["year"], single_result["estimate"]
    in_range = fit_min <= yr <= fit_max
    actual = working_df.loc[working_df.year == yr, "value"]
    st.write(f"**Year:** {yr} · **Estimate:** {val:.10g} {indicator_units} · **Classification:** {'within' if in_range else 'outside'} the fitted historical range ({fit_min}–{fit_max}).")
    if not actual.empty:
        st.caption(f"An observed value exists for this year: {float(actual.iloc[0]):.10g} {indicator_units}. The model estimate is a fitted value, not the same thing as the observation.")
    else:
        st.caption("No actual observation exists for this year in the working table.")

st.subheader("Batch year estimates")
batch_mode = st.radio("Batch input type", ["Year range", "Custom years"], horizontal=True)
batch_years = []
invalid_batch = []
if batch_mode == "Year range":
    bc1, bc2, bc3 = st.columns(3)
    batch_start = int(bc1.number_input("Start year", min_value=1950, max_value=2100, value=max(1950, fit_min), step=1, key="batch_start"))
    batch_end = int(bc2.number_input("End year", min_value=1950, max_value=2100, value=min(2100, fit_max), step=1, key="batch_end"))
    batch_step = int(bc3.slider("Range step", 1, 10, 1, key="batch_step"))
    if batch_start > batch_end:
        invalid_batch.append("Start year must be less than or equal to end year.")
    else:
        batch_years = list(range(batch_start, batch_end + 1, batch_step))
else:
    custom_text = st.text_input("Comma-separated years", value=f"{fit_min},{fit_max}", placeholder="e.g. 1980, 1990, 2025")
    for token in custom_text.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            num = float(token)
            if not num.is_integer() or not 1950 <= num <= 2100:
                invalid_batch.append(f"'{token}' must be an integer year from 1950 through 2100.")
            else:
                batch_years.append(int(num))
        except ValueError:
            invalid_batch.append(f"'{token}' is not a valid year.")
    batch_years = sorted(set(batch_years))
for message in invalid_batch:
    st.error(message)
if st.button("Calculate batch", disabled=bool(invalid_batch) or not batch_years):
    batch_records = []
    for yr in batch_years:
        actual_rows = working_df.loc[working_df.year == yr, "value"]
        estimate = predict_poly(model, yr)
        batch_records.append({"year": yr, "estimate": estimate, "units": indicator_units,
                              "range status": "within fitted range" if fit_min <= yr <= fit_max else "outside fitted range",
                              "actual observation": float(actual_rows.iloc[0]) if not actual_rows.empty else np.nan})
    st.session_state["calculated_results"] = pd.DataFrame(batch_records)
    st.session_state["batch_fingerprint"] = fingerprint
results_df = st.session_state.get("calculated_results", pd.DataFrame())
if not results_df.empty and st.session_state.get("batch_fingerprint") == fingerprint:
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    st.caption("Batch and single-year values use the same centered polynomial transformation; no second transformation is applied.")
    batch_points = results_df[["year", "estimate"]].dropna()
    fig_batch = make_plot(plot_years, plot_vals, "Model with requested batch estimates", "Year", f"Estimate ({indicator_units})",
                          points=(working_df.year.to_numpy(), working_df.value.to_numpy()), point_labels="Observed data",
                          curves=[(plot_years, plot_vals, "Polynomial fit / extrapolation", "-")])
    fig_batch.axes[0].scatter(batch_points.year, batch_points.estimate, marker="D", s=55, label="Batch estimates")
    fig_batch.axes[0].legend(fontsize=8)
    st.pyplot(fig_batch, use_container_width=True)
    plt.close(fig_batch)

st.subheader("Mathematics pathway interpretation")
if pathway == "Honors Algebra 2":
    st.write(f"- **Domain:** the polynomial is mathematically defined for all real years, but evidence-based interpretation is strongest over {fit_min}–{fit_max}.")
    st.write(f"- **Range:** the plotted model values run from {np.nanmin(plot_vals):.5g} to {np.nanmax(plot_vals):.5g} over the displayed interval; this is not the global range.")
    st.write("- **End behavior and trends:** high-degree polynomial end behavior is controlled by its leading coefficient and degree; far-future growth can become unrealistic.")
    st.write(f"- **Average rate:** over the observed interval, the model's average change is {(predict_poly(model, fit_max)-predict_poly(model, fit_min))/(fit_max-fit_min):.5g} {indicator_units} per year (when the interval has positive length).")
elif pathway == "Precalculus":
    st.write(f"- **Degree/coefficient effects:** degree {degree} allows up to {degree-1} turning points; higher-degree terms can create bends and unstable extrapolation.")
    st.write("- **Turning points and concavity:** changing average rates suggest where the graph changes direction or bends; these are properties of the fitted model, not proof of historical causes.")
    st.write(f"- **Average rate across the observed interval:** {(predict_poly(model, fit_max)-predict_poly(model, fit_min))/(fit_max-fit_min):.5g} {indicator_units} per year.")
else:
    st.write("The derivative-based analysis below uses a separate student-selected function, which can be the historical polynomial or another contextual function.")
st.warning("Polynomial extrapolation is not a validated 95% prediction interval. Residuals and MAE describe historical fit error; they do not capture all structural changes, measurement uncertainty, policy changes, or future shocks.")

st.header("2. Contextual function lab")
st.write("Explore a model independent of the historical fit, or use the fitted polynomial as a function. Contextual examples are illustrative unless their parameters come from a justified source.")
context = st.selectbox("Context", list(CONTEXTS))
suggested_y, suggested_y_unit, suggested_x, suggested_x_unit, suggested_description, suggested_type = CONTEXTS[context]
context_description = st.text_input("Context description", value=suggested_description, key="context_description")
c1, c2, c3, c4 = st.columns(4)
y_name = c1.text_input("Dependent variable", value=suggested_y, key="y_name")
y_unit = c2.text_input("Dependent units", value=suggested_y_unit, key="y_unit")
x_name = c3.text_input("Independent variable", value=suggested_x, key="x_name")
x_unit = c4.text_input("Independent units", value=suggested_x_unit, key="x_unit")
function_source = st.radio("Function source", ["Historical curve of best fit", "Suggested function", "Build from parameters", "Enter my own function"], horizontal=False)
if function_source == "Historical curve of best fit":
    # A polynomial in x measured as calendar year is exposed as the fitted model function.
    x_symbol = SYMBOLS["x"]
    expr = sum(float(c) * (x_symbol - model["base"])**i for i, c in enumerate(model["coef"]))
    function_label = equation.replace("t", f"({x_name} − {model['base']})")
    default_interval = (float(fit_min), float(fit_max))
elif function_source == "Suggested function":
    function_type = st.selectbox("Suggested function family", list(FUNCTION_PRESETS), index=list(FUNCTION_PRESETS).index({
        "polynomial":"Polynomial", "exponential":"Exponential", "trigonometric":"Trigonometric"}.get(suggested_type, "Polynomial")))
    preset_text = FUNCTION_PRESETS[function_type]
    pa, pb, pc, ph = st.columns(4)
    params = {"a": pa.number_input("a", value=PARAM_DEFAULTS["a"], key=f"pa_{context}"),
              "b": pb.number_input("b", value=PARAM_DEFAULTS["b"], key=f"pb_{context}"),
              "c": pc.number_input("c", value=PARAM_DEFAULTS["c"], key=f"pc_{context}"),
              "h": ph.number_input("h", value=PARAM_DEFAULTS["h"], key=f"ph_{context}")}
    st.code(preset_text, language="text")
    expr = safe_parse_function(preset_text, params)
    function_label = str(expr)
    default_interval = (-10.0, 10.0)
elif function_source == "Build from parameters":
    family = st.selectbox("Function family", list(FUNCTION_PRESETS))
    pa, pb, pc, ph = st.columns(4)
    params = {"a": pa.number_input("a", value=1.0, key="build_a"), "b": pb.number_input("b", value=.2, key="build_b"),
              "c": pc.number_input("c", value=0.0, key="build_c"), "h": ph.number_input("h", value=-1.0, key="build_h")}
    template = FUNCTION_PRESETS[family]
    st.caption(f"Template: {template}")
    expr = safe_parse_function(template, params)
    function_label = str(expr)
    default_interval = (-10.0, 10.0)
else:
    user_expression = st.text_input("Enter f(x)", value="x**3 - 3*x", help="Examples: x^2+2*x+1, (x+1)/(x-2), exp(0.2*x), log(x+3), sin(x).")
    try:
        expr = safe_parse_function(user_expression, {})
        function_label = f"f(x) = {sp.sstr(expr)}"
        default_interval = (-5.0, 5.0)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

if function_source == "Historical curve of best fit":
    st.caption("The independent variable x is calendar year for this option. For other function sources, x is the input in your chosen units.")
else:
    st.caption("Function parameters are illustrative unless explicitly tied to observed or cited evidence.")
try:
    derivative = sp.diff(expr, SYMBOLS["x"])
    second_derivative = sp.diff(derivative, SYMBOLS["x"])
    st.markdown(f"**Function:** \\(f(x) = {sp.latex(expr)}\\)")
    if pathway == "Calculus":
        st.markdown(f"**First derivative:** \\(f'(x) = {sp.latex(derivative)}\\)")
        st.markdown(f"**Second derivative:** \\(f''(x) = {sp.latex(second_derivative)}\\)")
except Exception as exc:
    st.error(f"Could not differentiate this function: {exc}")
    st.stop()

x1, x2, x3 = st.columns(3)
interval_min = x1.number_input("Analysis interval start", value=float(default_interval[0]), key=f"interval_min_{context}_{function_source}")
interval_max = x2.number_input("Analysis interval end", value=float(default_interval[1]), key=f"interval_max_{context}_{function_source}")
if interval_min >= interval_max:
    st.error("Analysis interval start must be less than the end.")
    st.stop()
current_x = x3.slider("Current input x", min_value=float(interval_min), max_value=float(interval_max),
                      value=float((interval_min+interval_max)/2), step=max((interval_max-interval_min)/200, 0.01), key=f"current_x_{context}_{function_source}")
if pathway == "Calculus":
    st.subheader("Interactive calculus analysis")
    show_f = st.checkbox("Show f(x)", True)
    show_d1 = st.checkbox("Show f′(x)", True)
    show_d2 = st.checkbox("Show f″(x)", False)
    a1, a2 = st.columns(2)
    sec_start = a1.slider("Secant interval start", float(interval_min), float(interval_max), float(interval_min), key=f"sec_start_{context}_{function_source}")
    sec_end = a2.slider("Secant interval end", float(interval_min), float(interval_max), float(interval_max), key=f"sec_end_{context}_{function_source}")
else:
    show_f, show_d1, show_d2 = True, False, False
    sec_start, sec_end = float(interval_min), float(interval_max)

f_fn, d1_fn, d2_fn = lambdify_real(expr), lambdify_real(derivative), lambdify_real(second_derivative)
grid = np.linspace(float(interval_min), float(interval_max), 1200)
curve_list = []
if show_f: curve_list.append((grid, f_fn(grid), "f(x)", "-"))
if show_d1: curve_list.append((grid, d1_fn(grid), "f′(x)", "--"))
if show_d2: curve_list.append((grid, d2_fn(grid), "f″(x)", ":"))
if not curve_list: curve_list = [(grid, f_fn(grid), "f(x)", "-")]
fig_fun, ax = plt.subplots(figsize=(10, 5.2))
for xx, yy, label, style in curve_list:
    ax.plot(xx, yy, style, linewidth=2, label=label)
fcur = real_value(f_fn, current_x)
dcur = real_value(d1_fn, current_x)
ddcur = real_value(d2_fn, current_x)
if np.isfinite(fcur):
    ax.scatter([current_x], [fcur], marker="o", s=70, label=f"Current point ({current_x:.4g}, {fcur:.4g})", zorder=5)
    if np.isfinite(dcur):
        tangent_x = np.linspace(max(interval_min, current_x-(interval_max-interval_min)*.12), min(interval_max, current_x+(interval_max-interval_min)*.12), 80)
        ax.plot(tangent_x, fcur + dcur*(tangent_x-current_x), "--", label="Tangent line")
if np.isfinite(real_value(f_fn, sec_start)) and np.isfinite(real_value(f_fn, sec_end)) and sec_end != sec_start:
    ya, yb = real_value(f_fn, sec_start), real_value(f_fn, sec_end)
    slope = (yb-ya)/(sec_end-sec_start)
    sx = np.linspace(sec_start, sec_end, 80)
    ax.plot(sx, ya+slope*(sx-sec_start), ":", linewidth=2, label="Secant line")
ax.axhline(0, color="black", linewidth=.6, alpha=.5)
ax.set_title(f"{context}: {context_description}")
ax.set_xlabel(f"{x_name} ({x_unit})")
ax.set_ylabel(f"{y_name} ({y_unit})")
ax.grid(True, alpha=.25)
ax.legend(fontsize=8, loc="best")
fig_fun.tight_layout()
st.pyplot(fig_fun, use_container_width=True)
fun_png = io.BytesIO()
fig_fun.savefig(fun_png, format="png", dpi=140, bbox_inches="tight")
fun_png_bytes = fun_png.getvalue()
plt.close(fig_fun)

v1, v2, v3 = st.columns(3)
v1.metric(f"f(x) · {y_unit}", f"{fcur:.7g}" if np.isfinite(fcur) else "undefined")
v2.metric(f"f′(x) · {y_unit}/{x_unit}", f"{dcur:.7g}" if np.isfinite(dcur) else "undefined")
v3.metric(f"f″(x) · {y_unit}/{x_unit}²", f"{ddcur:.7g}" if np.isfinite(ddcur) else "undefined")
if np.isfinite(dcur):
    rate_text = "increasing" if dcur > 1e-8 else ("decreasing" if dcur < -1e-8 else "locally flat")
    st.write(f"At x = {current_x:.5g}, the function is {fcur:.5g} {y_unit}; its instantaneous rate is {dcur:.5g} {y_unit} per {x_unit} ({rate_text}). The second derivative is {ddcur:.5g} {y_unit} per {x_unit}² and describes how that rate changes.")
else:
    st.write("The function or derivative is undefined at this input, so a numerical rate interpretation is not available.")
if sec_start != sec_end:
    fs, fe = real_value(f_fn, sec_start), real_value(f_fn, sec_end)
    if np.isfinite(fs) and np.isfinite(fe):
        avg_rate = (fe-fs)/(sec_end-sec_start)
        st.write(f"Average rate of change from x = {sec_start:.5g} to x = {sec_end:.5g}: **{avg_rate:.7g} {y_unit} per {x_unit}**.")
    else:
        st.warning("The selected secant interval includes an undefined endpoint; average rate of change is unavailable.")

if pathway == "Calculus":
    st.subheader("Critical points, extrema, concavity, and domain")
    # Numerical roots are approximations, with sign tests on either side.
    crit_candidates = numeric_roots(derivative, float(interval_min), float(interval_max))
    second_roots = numeric_roots(second_derivative, float(interval_min), float(interval_max))
    critical_rows = []
    for root in crit_candidates:
        val = real_value(f_fn, root)
        delta = max((interval_max-interval_min)*1e-4, 1e-5)
        left_d, right_d = real_value(d1_fn, root-delta), real_value(d1_fn, root+delta)
        classification = "stationary, no sign-change classification"
        if np.isfinite(left_d) and np.isfinite(right_d):
            if left_d > 0 and right_d < 0: classification = "local maximum (numerical first-derivative sign test)"
            elif left_d < 0 and right_d > 0: classification = "local minimum (numerical first-derivative sign test)"
        critical_rows.append({"x (approx.)": root, "f(x)": val, "classification": classification})
    if critical_rows:
        st.dataframe(pd.DataFrame(critical_rows), use_container_width=True, hide_index=True)
    else:
        st.write("No sign-changing roots of f′(x) were detected in the selected interval. Even-multiplicity roots may not be detected by a sign-change scan.")
    concavity_rows = []
    for root in second_roots:
        delta = max((interval_max-interval_min)*1e-4, 1e-5)
        left_dd, right_dd = real_value(d2_fn, root-delta), real_value(d2_fn, root+delta)
        changes = np.isfinite(left_dd) and np.isfinite(right_dd) and left_dd * right_dd < 0
        concavity_rows.append({"x (approx.)": root, "f(x)": real_value(f_fn, root),
                               "concavity changes?": "yes (numerically checked)" if changes else "not confirmed"})
    if concavity_rows:
        st.write("Candidate inflection points (a zero of f″ is only a candidate until concavity changes):")
        st.dataframe(pd.DataFrame(concavity_rows), use_container_width=True, hide_index=True)
    # Absolute extrema over sampled, finite points plus endpoints.
    sample_x = np.linspace(interval_min, interval_max, 8000)
    sample_y = f_fn(sample_x)
    valid = np.isfinite(sample_y)
    if valid.any():
        finite_indices = np.where(valid)[0]
        imin, imax = finite_indices[np.argmin(sample_y[valid])], finite_indices[np.argmax(sample_y[valid])]
        st.write(f"Numerical absolute minimum on sampled finite points: x ≈ {sample_x[imin]:.7g}, f(x) ≈ {sample_y[imin]:.7g}; maximum: x ≈ {sample_x[imax]:.7g}, f(x) ≈ {sample_y[imax]:.7g}. Endpoints are included when defined; these are numerical approximations, not symbolic proofs.")
    try:
        domain = sp.calculus.util.continuous_domain(expr, SYMBOLS["x"], sp.Interval(float(interval_min), float(interval_max)))
        st.write(f"Symbolic continuous domain within the selected interval: `{domain}`")
        if expr.is_rational_function(SYMBOLS["x"]):
            den = sp.denom(sp.together(expr))
            singularities = sp.solve(sp.Eq(den, 0), SYMBOLS["x"])
            in_interval = [str(s) for s in singularities if s.is_real and float(interval_min) <= float(s) <= float(interval_max)]
            st.write("Potential rational discontinuities in interval: " + (", ".join(in_interval) if in_interval else "none detected"))
    except Exception:
        st.caption("Could not fully characterize the symbolic domain; inspect non-finite gaps in the graph.")
    st.caption("Critical points and inflection candidates are numerical approximations. A zero of f′ is not automatically an extremum, and a zero of f″ is not automatically an inflection point. Root scanning can miss tangential roots or closely spaced features.")
elif pathway == "Precalculus":
    st.write("Changing average rates can indicate turning points and concavity. The plotted function and sampled rates are approximations on the selected interval; the app does not infer real-world causes.")
else:
    st.write("Use the graph to describe domain, range on the selected interval, end behavior, intercepts, and average rates. Model validity may be narrower than the mathematical domain.")

# Lightweight synchronized playback: the moving point and all displayed values use current_x.
with st.expander("Simulation controls"):
    visual = st.selectbox("Contextual visual", ["Walking person with prosthetic leg", "Moving object / projectile",
        "Tank or drug-concentration indicator", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
    speed = st.select_slider("Playback speed", options=["Slow", "Medium", "Fast"], value="Medium")
    play_col, pause_col, reset_col = st.columns(3)
    if "playback" not in st.session_state: st.session_state["playback"] = False
    if play_col.button("▶ Play"): st.session_state["playback"] = True
    if pause_col.button("⏸ Pause"): st.session_state["playback"] = False
    if reset_col.button("↺ Reset"):
        st.session_state["playback"] = False
        st.session_state[f"current_x_{context}_{function_source}"] = float((interval_min+interval_max)/2)
        st.rerun()
    st.caption("Playback advances the input slider on reruns; graph point, tangent, rates, and explanation are synchronized. Visual scaling is illustrative and normalized to the current analysis interval.")
    # A single controlled step per rerun avoids an unbounded blocking animation loop.
    if st.session_state["playback"]:
        step_size = (interval_max-interval_min) / (100 if speed == "Slow" else (35 if speed == "Medium" else 12))
        next_x = current_x + step_size
        if next_x > interval_max: next_x = interval_min
        st.session_state[f"current_x_{context}_{function_source}"] = float(next_x)
        time.sleep(0.15 if speed == "Fast" else (0.35 if speed == "Medium" else 0.7))
        st.rerun()
    # Simple responsive schematic rather than a claim of biomechanical/medical fidelity.
    normalized = 0.5 if not np.isfinite(fcur) else float(np.clip((fcur - np.nanmin(f_fn(grid))) / max(np.nanmax(f_fn(grid))-np.nanmin(f_fn(grid)), 1e-9), 0, 1))
    symbols = {"Walking person with prosthetic leg":"🚶 🦿", "Moving object / projectile":"⚽", "Tank or drug-concentration indicator":"🧪",
               "Population / cell colony":"🦠", "Thermometer / weather":"🌡️", "Business / economic indicator":"📊"}
    st.markdown(f"<div style='font-size:2.4rem;text-align:center'>{symbols[visual]}</div>", unsafe_allow_html=True)
    st.progress(normalized, text=f"Illustrative scaled level: {normalized:.0%} · actual model value {fcur:.5g} {y_unit}" if np.isfinite(fcur) else "Function undefined at current input")
    st.caption("The icon is a contextual metaphor, not a physical simulation. Its level maps the function value to the finite plotted range; it does not model anatomy, dosage, weather dynamics, or physical laws.")

st.header("3. Country comparison")
st.write("Comparison uses the same indicator and World Bank source. Country coverage may differ because the World Bank series contain gaps or different reporting histories.")
comparison_series = []
for label in comparison_labels:
    ccode = COUNTRIES[label]
    comp_df, comp_status = fetch_indicator(ccode, indicator_code, start_year, end_year)
    if comp_df.empty:
        st.warning(f"{label}: {comp_status}")
        continue
    comparison_series.append((label, comp_df))
if comparison_series:
    fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4.8))
    ax_cmp.plot(working_df.year, working_df.value, marker="o", linewidth=2, label=f"{country_label} (working data)")
    for label, df in comparison_series:
        ax_cmp.plot(df.year, df.value, marker=".", linewidth=1.4, label=label)
    ax_cmp.set_title(f"{indicator_name}: country comparison")
    ax_cmp.set_xlabel("Year")
    ax_cmp.set_ylabel(f"{indicator_name} ({indicator_units})")
    ax_cmp.grid(True, alpha=.25)
    ax_cmp.legend(fontsize=8)
    fig_cmp.tight_layout()
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)
    coverage = [{"country": country_label, "first year": int(working_df.year.min()), "last year": int(working_df.year.max()), "observations": len(working_df)}]
    coverage += [{"country": label, "first year": int(df.year.min()), "last year": int(df.year.max()), "observations": len(df)} for label, df in comparison_series]
    st.dataframe(pd.DataFrame(coverage), use_container_width=True, hide_index=True)
else:
    st.info("Select one or more comparison countries to display a comparison.")

st.header("4. Export the current analysis")
export_results = results_df.copy() if not results_df.empty and st.session_state.get("batch_fingerprint") == fingerprint else pd.DataFrame(columns=["year","estimate","units","range status","actual observation"])
summary = pd.DataFrame([{
    "app": APP_TITLE, "retrieved_at_local": retrieved_at, "country": country_label, "country_code": country_code,
    "indicator": indicator_name, "indicator_code": indicator_code, "units": indicator_units,
    "definition": indicator_definition, "requested_start": start_year, "requested_end": end_year,
    "actual_first_year": fit_min, "actual_last_year": fit_max, "observation_count": model["n"],
    "data_status": "user-modified scenario" if is_modified else "source data",
    "degree": degree, "base_year": model["base"], "equation": equation,
    "r_squared_in_sample": model["r2"], "mae_in_sample": model["mae"], "mean": model["mean"],
    "limitation": "Polynomial extrapolations are not validated prediction intervals; MAE is in-sample."
}])
csv_bytes = working_df.to_csv(index=False).encode("utf-8")
xlsx_buffer = io.BytesIO()
with pd.ExcelWriter(xlsx_buffer, engine="xlsxwriter") as writer:
    summary.to_excel(writer, sheet_name="Summary", index=False)
    working_df.to_excel(writer, sheet_name="Working data", index=False)
    export_results.to_excel(writer, sheet_name="Calculated years", index=False)
    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1, "text_wrap": True})
    for sheet_name, frame in [("Summary", summary), ("Working data", working_df), ("Calculated years", export_results)]:
        ws = writer.sheets[sheet_name]
        ws.freeze_panes(1, 0)
        if len(frame.columns):
            ws.autofilter(0, 0, max(len(frame), 1), len(frame.columns)-1)
            for col_num, col_name in enumerate(frame.columns):
                ws.write(0, col_num, col_name, header_fmt)
                max_len = max(12, min(42, len(str(col_name))+2))
                ws.set_column(col_num, col_num, max_len)
xlsx_bytes = xlsx_buffer.getvalue()
pdf_meta = {"lines": [
    f"Country: {country_label} ({country_code}) · Indicator: {indicator_name} ({indicator_code})",
    f"Definition: {indicator_definition}",
    f"Units: {indicator_units} · Requested period: {start_year}–{end_year} · Retrieved: {retrieved_at}",
    f"Actual coverage: {fit_min}–{fit_max}, n={model['n']} · Data status: {'USER-MODIFIED SCENARIO' if is_modified else 'source data'}",
    f"Model: degree {degree}, base year {model['base']} · {equation}",
    f"In-sample R²={model['r2']:.7g}; in-sample MAE={model['mae']:.7g} {indicator_units}; mean={model['mean']:.7g}",
    "Limitations: extrapolation is not a validated prediction interval. The model does not identify causes. MAE/residuals summarize in-sample fit only.",
    f"Function lab context: {context} · f(x)={sp.sstr(expr)}; x is {x_name} ({x_unit}), y is {y_name} ({y_unit}).",
    f"World Bank source: https://data.worldbank.org/indicator/{indicator_code}",
]}
pdf_bytes = make_pdf(pdf_meta, fun_png_bytes, working_df, export_results)
d1, d2, d3 = st.columns(3)
d1.download_button("Download working data CSV", data=csv_bytes, file_name="working_data.csv", mime="text/csv", use_container_width=True)
d2.download_button("Download Excel workbook", data=xlsx_bytes, file_name="hispanic_heritage_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
d3.download_button("Download PDF report", data=pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf", use_container_width=True)

if printer_mode:
    st.divider()
    st.subheader("Printer-friendly summary")
    st.markdown(f"**{country_label} — {indicator_name}**  \nCoverage: {fit_min}–{fit_max} (n={model['n']}) · Degree {degree} · {equation}  \nIn-sample R²: {model['r2']:.5f} · In-sample MAE: {model['mae']:.5g} {indicator_units} · Data: {'user-modified scenario' if is_modified else 'source'}")
    st.write(indicator_definition)
    st.write("Interpretation must distinguish observations from model estimates. Extrapolation is uncertain and is not a validated 95% prediction interval.")
    st.dataframe(working_df.head(20), use_container_width=True, hide_index=True)

with st.expander("How the app works and what to defend in a presentation"):
    st.markdown("""
    **Data flow:** the app requests one official World Bank indicator through its JSON API; it retains a source table and a separate editable working table. Edits are validated before modeling. Comparison countries are fetched independently, so their coverage may differ.

    **Polynomial model:** with base year \(b\), the model uses \(t=year-b\) and least-squares polynomial coefficients. R² and MAE are in-sample metrics. A chronological holdout check, when available, fits earlier observations and scores later ones. None of these metrics guarantees future accuracy.

    **Function lab:** SymPy parses a restricted expression vocabulary, substitutes user-controlled parameters, and differentiates symbolically. Numerical root scans approximate critical and inflection candidates; sign checks are used before classifying them. Numerical scans can miss tangential roots and close features. Rational/logarithmic functions may be undefined at some inputs, which appear as gaps.

    **Interpretation:** f(x) has dependent-variable units; f′(x) has dependent units per independent unit; f″(x) has dependent units per independent-unit squared. If position is measured in meters and time in seconds, these correspond to meters, m/s, and m/s². Other contexts depend on the variables and model assumptions.

    **Limitations:** no causes are inferred from a curve; World Bank observations can be missing or revised; current-dollar indicators are not inflation-adjusted unless the series says so; polynomial extrapolation can become implausible. The contextual icons are schematic and not clinical, biomechanical, or engineering simulators.
    """)
st.caption("Source: World Bank API and indicator metadata. Retrieval timestamp reflects this app session; API cache lifetime is one hour.")
