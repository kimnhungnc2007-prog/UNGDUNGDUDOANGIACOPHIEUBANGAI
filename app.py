import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ================= CẤU HÌNH =================
st.set_page_config(page_title="AI dự đoán cổ phiếu", page_icon="📈", layout="wide")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

st.title("📈 AI DỰ ĐOÁN GIÁ CỔ PHIẾU")
st.caption("Sử dụng Deep Learning LSTM và dữ liệu Yahoo Finance")

# ================= HÀM XỬ LÝ =================
@st.cache_data(ttl=1800)
def download_data(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [c for c in df.columns if c[0] == "Close"]
        if not close_cols:
            return pd.DataFrame()
        close = df[close_cols[0]]
    else:
        if "Close" not in df.columns:
            return pd.DataFrame()
        close = df["Close"]

    result = pd.DataFrame({"Close": close})
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    return result.dropna().sort_index()


def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def create_model(window, units):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(units, return_sequences=True),
        LSTM(units),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# ================= GIAO DIỆN =================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    ticker = st.text_input("Mã cổ phiếu", "VNM.VN").strip().upper()
    start_date = st.date_input("Từ ngày", date(2018, 1, 1))
    end_date = st.date_input("Đến ngày", date.today())
    window = st.slider("Số phiên đầu vào", 20, 120, 60, 10)
    units = st.selectbox("Neuron LSTM", [16, 32, 50, 64, 100], index=2)
    epochs = st.slider("Số epoch", 5, 50, 15)
    batch_size = st.selectbox("Batch size", [16, 32, 64], index=1)
    forecast_days = st.slider("Số phiên dự báo", 1, 30, 10)
    run = st.button("🚀 BẮT ĐẦU DỰ BÁO", type="primary")

st.info("Ví dụ mã cổ phiếu Việt Nam: VNM.VN, FPT.VN, HPG.VN, VCB.VN.")

if run:
    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    with st.spinner("Đang tải dữ liệu..."):
        try:
            df = download_data(
                ticker,
                start_date.isoformat(),
                (end_date + timedelta(days=1)).isoformat()
            )
        except Exception as e:
            st.error(f"Lỗi tải dữ liệu: {e}")
            st.stop()

    if df.empty:
        st.error("Không có dữ liệu. Kiểm tra mã cổ phiếu hoặc kết nối Internet.")
        st.stop()

    data = df[["Close"]].values.astype(float)

    if len(data) < window + 50:
        st.error("Dữ liệu không đủ. Hãy chọn khoảng thời gian dài hơn.")
        st.stop()

    # ================ CHIA DỮ LIỆU ================
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train_all, y_train_all = create_sequences(train_scaled, window)

    val_size = max(1, int(len(X_train_all) * 0.1))
    X_train = X_train_all[:-val_size]
    y_train = y_train_all[:-val_size]
    X_val = X_train_all[-val_size:]
    y_val = y_train_all[-val_size:]

    test_combined = np.vstack([train_scaled[-window:], test_scaled])
    X_test, y_test = create_sequences(test_combined, window)

    X_train = X_train.reshape(-1, window, 1)
    X_val = X_val.reshape(-1, window, 1)
    X_test = X_test.reshape(-1, window, 1)

    # ================ HUẤN LUYỆN ================
    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    model = create_model(window, units)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    with st.spinner("AI đang học dữ liệu..."):
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

    # ================ ĐÁNH GIÁ ================
    pred_scaled = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(pred_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mae = mean_absolute_error(actual, predictions)
    r2 = r2_score(actual, predictions) if len(actual) > 1 else np.nan

    latest_price = float(data[-1, 0])
    previous_price = float(data[-2, 0])
    change = (latest_price / previous_price - 1) * 100

    st.subheader(f"Kết quả phân tích {ticker}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Giá gần nhất", f"{latest_price:,.2f}", f"{change:+.2f}%")
    c2.metric("RMSE", f"{rmse:,.4f}")
    c3.metric("MAE", f"{mae:,.4f}")
    c4.metric("R²", f"{r2:.4f}" if np.isfinite(r2) else "N/A")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Giá lịch sử",
        "Đánh giá AI",
        "Dự báo tương lai",
        "Dữ liệu"
    ])

    # ================ BIỂU ĐỒ GIÁ ================
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["Close"], label="Giá đóng cửa")
        ax.set_title(f"Lịch sử giá {ticker}")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ================ BIỂU ĐỒ ĐÁNH GIÁ ================
    with tab2:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index[train_size:], actual.flatten(), label="Giá thực tế")
        ax.plot(df.index[train_size:], predictions.flatten(), label="Giá dự báo")
        ax.set_title("Giá thực tế và giá AI dự báo trên tập kiểm tra")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.history["loss"], label="Train Loss")
        ax.plot(history.history["val_loss"], label="Validation Loss")
        ax.set_title("Quá trình huấn luyện")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "RMSE và MAE đo sai số dự báo; giá trị thấp thường tốt hơn. "
            "R² có thể âm nếu mô hình dự báo kém. Các chỉ số quá khứ "
            "không đảm bảo độ chính xác trong tương lai."
        )

    # ================ DỰ BÁO TƯƠNG LAI ================
    # Fit lại scaler và huấn luyện mô hình trên toàn bộ dữ liệu
    # sau khi đã đánh giá mô hình bằng tập test.
    full_scaler = MinMaxScaler()
    full_scaled = full_scaler.fit_transform(data)

    X_full, y_full = create_sequences(full_scaled, window)
    X_full = X_full.reshape(-1, window, 1)

    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    future_model = create_model(window, units)
    future_model.fit(
        X_full,
        y_full,
        epochs=len(history.history["loss"]),
        batch_size=batch_size,
        shuffle=False,
        verbose=0
    )

    current_window = full_scaled[-window:].reshape(1, window, 1)
    future_scaled = []

    for _ in range(forecast_days):
        next_value = future_model.predict(current_window, verbose=0)[0, 0]
        future_scaled.append(next_value)

        current_window = np.concatenate(
            [
                current_window[:, 1:, :],
                np.array(next_value).reshape(1, 1, 1)
            ],
            axis=1
        )

    future_prices = full_scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    future_dates = pd.bdate_range(
        start=pd.Timestamp(df.index[-1]) + pd.offsets.BDay(1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "Ngày dự báo": future_dates.strftime("%d/%m/%Y"),
        "Giá dự báo": future_prices
    })

    with tab3:
        st.subheader(f"Dự báo {forecast_days} phiên tiếp theo")

        st.dataframe(
            forecast_df.style.format({"Giá dự báo": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Tải kết quả CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        recent = df.tail(120)
        ax.plot(recent.index, recent["Close"], label="Giá lịch sử")
        ax.plot(
            future_dates,
            future_prices,
            marker="o",
            linestyle="--",
            label="Giá dự báo"
        )
        ax.set_title(f"Dự báo giá cổ phiếu {ticker}")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.warning(
            "Dự báo nhiều phiên dùng giá dự báo trước làm đầu vào cho phiên sau, "
            "nên sai số có thể tích lũy. Kết quả chỉ phục vụ tham khảo, "
            "không phải cam kết lợi nhuận hay khuyến nghị mua bán."
        )

    # ================ BẢNG DỮ LIỆU ================
    with tab4:
        st.dataframe(df.tail(200), use_container_width=True)

        st.download_button(
            "⬇️ Tải dữ liệu lịch sử",
            data=df.to_csv(index=True).encode("utf-8-sig"),
            file_name=f"lich_su_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )
else:
    st.markdown("""
    ### Hướng dẫn
    1. Nhập mã cổ phiếu ở thanh bên.
    2. Chọn thời gian lấy dữ liệu.
    3. Chọn số phiên muốn dự báo.
    4. Nhấn **BẮT ĐẦU DỰ BÁO**.
    """)
