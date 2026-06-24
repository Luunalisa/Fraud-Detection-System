import os
import time

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛡️ Fraud Detection")
    st.caption(f"API: `{API_URL}`")

    try:
        info = requests.get(f"{API_URL}/info", timeout=5).json()
        st.success("API connected")
        st.metric("Model version", info["model_version"])
        st.metric("Threshold",     f"{info['threshold']:.4f}")
        st.metric("Features",      info["feature_count"])

        with st.expander("Model metrics"):
            m = info.get("metrics", {})
            col1, col2 = st.columns(2)
            col1.metric("ROC-AUC",   f"{m.get('test_roc_auc', 0):.4f}")
            col1.metric("AUPRC",     f"{m.get('test_auprc', 0):.4f}")
            col2.metric("Precision", f"{m.get('test_precision', 0):.4f}")
            col2.metric("Recall",    f"{m.get('test_recall', 0):.4f}")
    except Exception:
        st.error("API unreachable")

    st.divider()
    mode = st.radio("Mode", ["Single transaction", "Batch scoring", "API Health"])


# ── Single transaction ────────────────────────────────────────────
if mode == "Single transaction":
    st.header("Score a single transaction")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount (USD)", min_value=0.0, value=149.62)
            time_  = st.number_input("Time", min_value=0.0, value=0.0)

        st.subheader("V1 – V14")
        cols = st.columns(7)
        defaults = [
            -1.3598, -0.0728,  2.5363,  1.3782, -0.3383,  0.4624,  0.2396,
             0.0987,  0.3638,  0.0908, -0.5516, -0.6178, -0.9914, -0.3112,
             1.4682, -0.4704,  0.2080,  0.0258,  0.4040,  0.2514,
            -0.0183,  0.2778, -0.1105,  0.0669,  0.1285, -0.1891,  0.1336, -0.0211,
        ]
        v_vals = {}
        for i in range(1, 15):
            v_vals[f"V{i}"] = cols[(i-1) % 7].number_input(
                f"V{i}", value=defaults[i-1], format="%.4f", key=f"v{i}"
            )

        st.subheader("V15 – V28")
        cols2 = st.columns(7)
        for i in range(15, 29):
            v_vals[f"V{i}"] = cols2[(i-15) % 7].number_input(
                f"V{i}", value=defaults[i-1], format="%.4f", key=f"v{i}"
            )

        submitted = st.form_submit_button("Score transaction", use_container_width=True)

    if submitted:
        payload = {**v_vals, "Amount": amount, "Time": time_}
        with st.spinner("Scoring..."):
            try:
                t0 = time.perf_counter()
                resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                latency = (time.perf_counter() - t0) * 1000
                resp.raise_for_status()
                result = resp.json()

                st.divider()
                col1, col2, col3, col4 = st.columns(4)

                if result["is_fraud"]:
                    col1.error("FRAUD DETECTED")
                else:
                    col1.success("LEGITIMATE")

                col2.metric("Probability", f"{result['probability']:.4%}")
                col3.metric("Threshold",   f"{result['threshold']:.4f}")
                col4.metric("Latency",     f"{latency:.0f} ms")

                st.progress(result["probability"])

                with st.expander("Raw response"):
                    st.json(result)

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API")
            except Exception as e:
                st.error(f"Error: {e}")


# ── Batch scoring ─────────────────────────────────────────────────
elif mode == "Batch scoring":
    st.header("Batch transaction scoring")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(10), use_container_width=True)

        required_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if st.button("Score all transactions", use_container_width=True):
                all_preds = []
                all_probs = []
                progress  = st.progress(0)
                batch_size = 1000
                n_batches  = (len(df) + batch_size - 1) // batch_size

                with st.spinner(f"Scoring {len(df):,} transactions..."):
                    t0 = time.perf_counter()
                    for i in range(n_batches):
                        chunk   = df.iloc[i*batch_size:(i+1)*batch_size]
                        payload = {"transactions": chunk[required_cols].to_dict(orient="records")}
                        resp    = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=60)
                        resp.raise_for_status()
                        result  = resp.json()
                        all_preds.extend(result["predictions"])
                        all_probs.extend(result["probabilities"])
                        progress.progress((i + 1) / n_batches)
                    latency = (time.perf_counter() - t0) * 1000

                df["prediction"]  = all_preds
                df["probability"] = all_probs
                df["is_fraud"]    = df["prediction"] == 1

                fraud_count = df["is_fraud"].sum()
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total",       f"{len(df):,}")
                col2.metric("Fraud",       f"{fraud_count:,}")
                col3.metric("Fraud rate",  f"{fraud_count/len(df):.2%}")
                col4.metric("Latency",     f"{latency:.0f} ms")

                st.dataframe(
                    df[["Amount", "prediction", "probability", "is_fraud"]],
                    use_container_width=True,
                )

                st.download_button(
                    "Download predictions CSV",
                    data=df.to_csv(index=False),
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ── API Health ────────────────────────────────────────────────────
elif mode == "API Health":
    st.header("API Health")

    if st.button("Refresh"):
        st.rerun()

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"Status: {health['status'].upper()}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Model loaded",  "Yes" if health["model_loaded"] else "No")
        col2.metric("Model version", health["model_version"])
        col3.metric("Threshold",     f"{health['threshold']:.4f}")

        st.subheader("Prometheus metrics")
        lines = [
            l for l in requests.get(f"{API_URL}/metrics", timeout=5).text.split("\n")
            if l.startswith("fraud_") and not l.startswith("#")
        ]
        if lines:
            st.code("\n".join(lines))
        else:
            st.info("No metrics yet — make some predictions first")

    except Exception as e:
        st.error(f"API unreachable: {e}")