import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
from datetime import date, timedelta

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# ==================================================
# 1. CẤU HÌNH STREAMLIT
# ==================================================

st.set_page_config(
    page_title="Dự báo giá cổ phiếu bằng LSTM",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ỨNG DỤNG DỰ BÁO GIÁ CỔ PHIẾU")
st.markdown("### Sử dụng mô hình trí tuệ nhân tạo LSTM")
st.write(
    "Ứng dụng phân tích dữ liệu giá cổ phiếu trong quá khứ, "
    "huấn luyện mô hình LSTM và dự báo giá trong tương lai."
)

# ==================================================
# 2. FIX SEED
# ==================================================

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)

WINDOW_SIZE = 60

# ==================================================
# 3. THANH ĐIỀU KHIỂN
# ==================================================

st.sidebar.header("⚙️ THIẾT LẬP MÔ HÌNH")

stock_list = {
    "Vinamilk - VNM": "VNM.VN",
    "FPT": "FPT.VN",
    "Hòa Phát - HPG": "HPG.VN",
    "Vietcombank - VCB": "VCB.VN",
    "Vingroup - VIC": "VIC.VN",
    "Thế Giới Di Động - MWG": "MWG.VN",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Nhập mã khác": "OTHER"
}

selected_stock = st.sidebar.selectbox(
    "Chọn mã cổ phiếu",
    list(stock_list.keys())
)

ticker = stock_list[selected_stock]

if ticker == "OTHER":
    ticker = st.sidebar.text_input(
        "Nhập mã cổ phiếu",
        "VNM.VN"
    ).strip().upper()

start_date = st.sidebar.date_input(
    "Ngày bắt đầu",
    value=date(2018, 1, 1)
)

end_date = st.sidebar.date_input(
    "Ngày kết thúc",
    value=date.today()
)

epochs = st.sidebar.slider(
    "Số vòng huấn luyện (Epochs)",
    min_value=5,
    max_value=100,
    value=10,
    step=5
)

batch_size = st.sidebar.selectbox(
    "Batch size",
    [16, 32, 64],
    index=1
)

future_days = st.sidebar.slider(
    "Số phiên dự báo tương lai",
    min_value=1,
    max_value=30,
    value=10
)

run = st.sidebar.button(
    "🚀 BẮT ĐẦU DỰ BÁO",
    type="primary",
    use_container_width=True
)

# ==================================================
# 4. HÀM TẢI DỮ LIỆU
# ==================================================

@st.cache_data(ttl=3600)
def load_stock_data(ticker, start_date, end_date):

    # end của yfinance là ngày không bao gồm
    end_download = (pd.Timestamp(end_date) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    df = yf.download(
        ticker,
        start=str(start_date),
        end=end_download,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df.empty:
        return df

    # Xử lý MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = df.dropna(subset=["Close"])

    return df


# ==================================================
# 5. TẠO CHUỖI DỮ LIỆU
# ==================================================

def create_sequences(data, window_size=60):

    X, y = [], []

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])
        y.append(data[i])

    return np.array(X), np.array(y)


# ==================================================
# 6. XÂY DỰNG MÔ HÌNH LSTM
# ==================================================

def create_model():

    model = Sequential()

    model.add(
        Input(shape=(60, 1))
    )

    model.add(
        LSTM(50, return_sequences=True)
    )

    model.add(
        LSTM(50)
    )

    model.add(
        Dense(1)
    )

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    return model


# ==================================================
# 7. HÀM VẼ BIỂU ĐỒ
# ==================================================

def show_chart(fig):
    st.pyplot(fig)
    plt.close(fig)


# ==================================================
# 8. CHẠY ỨNG DỤNG
# ==================================================

if run:

    if not ticker:
        st.error("Vui lòng nhập mã cổ phiếu.")
        st.stop()

    if start_date >= end_date:
        st.error("Ngày bắt đầu phải nhỏ hơn ngày kết thúc.")
        st.stop()

    # ----------------------------------------------
    # TẢI DỮ LIỆU
    # ----------------------------------------------

    with st.spinner("Đang tải dữ liệu từ Yahoo Finance..."):

        try:
            df = load_stock_data(
                ticker,
                start_date,
                end_date
            )
        except Exception as e:
            st.error(f"Lỗi tải dữ liệu: {e}")
            st.stop()

    if df.empty:
        st.error(
            "Không tìm thấy dữ liệu cổ phiếu. "
            "Vui lòng kiểm tra mã hoặc thời gian."
        )
        st.stop()

    if len(df) < 150:
        st.error(
            "Dữ liệu quá ít để huấn luyện LSTM. "
            "Vui lòng chọn khoảng thời gian dài hơn."
        )
        st.stop()

    st.success(
        f"Đã tải thành công {len(df)} phiên "
        f"giao dịch của {ticker}."
    )

    # ----------------------------------------------
    # THÔNG TIN CỔ PHIẾU# ----------------------------------------------

    st.header("1. THÔNG TIN CỔ PHIẾU")

    last_price = float(df["Close"].iloc[-1])
    previous_price = float(df["Close"].iloc[-2])

    price_change = (
        (last_price - previous_price)
        / previous_price * 100
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Mã cổ phiếu",
        ticker
    )

    col2.metric(
        "Giá đóng cửa gần nhất",
        f"{last_price:,.2f}"
    )

    col3.metric(
        "Thay đổi phiên gần nhất",
        f"{price_change:+.2f}%"
    )

    st.subheader("Biểu đồ giá cổ phiếu lịch sử")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df.index,
        df["Close"],
        color="blue",
        label="Giá đóng cửa"
    )

    ax.set_title(f"Lịch sử giá cổ phiếu {ticker}")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Giá")
    ax.legend()
    ax.grid(alpha=0.3)

    show_chart(fig)

    with st.expander("Xem dữ liệu gốc"):
        st.dataframe(df, use_container_width=True)

    # ----------------------------------------------
    # CHIA TRAIN / TEST
    # ----------------------------------------------

    st.header("2. CHUẨN BỊ DỮ LIỆU")

    data = df[["Close"]].values.astype("float32")

    train_size = int(0.8 * len(data))

    train_data = data[:train_size]
    test_data = data[train_size:]

    st.write("Tổng số phiên:", len(data))
    st.write("Số phiên Train:", len(train_data))
    st.write("Số phiên Test:", len(test_data))

    # ----------------------------------------------
    # CHUẨN HÓA DỮ LIỆU
    # ----------------------------------------------

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # ----------------------------------------------
    # TẠO SEQUENCES
    # ----------------------------------------------

    X_train, y_train = create_sequences(
        train_scaled,
        window_size=60
    )

    test_combined = np.vstack((
        train_scaled[-60:],
        test_scaled
    ))

    X_test, y_test = create_sequences(
        test_combined,
        window_size=60
    )

    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    X_test = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1],
        1
    )

    # ----------------------------------------------
    # TẠO VALIDATION
    # ----------------------------------------------

    val_size = max(1, int(0.1 * len(X_train)))

    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]

    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]

    if len(X_train_final) == 0:
        st.error("Không đủ dữ liệu huấn luyện.")
        st.stop()
# HUẤN LUYỆN MÔ HÌNH
    # ----------------------------------------------

    st.header("3. HUẤN LUYỆN MÔ HÌNH LSTM")

    with st.spinner(
        "Đang huấn luyện mô hình LSTM..."
    ):

        tf.keras.backend.clear_session()
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)

        model = create_model()

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        history = model.fit(
            X_train_final,
            y_train_final,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[early_stop],
            verbose=0
        )

    st.success("Huấn luyện mô hình hoàn tất!")

    st.write(
        "Số Epoch thực tế:",
        len(history.history["loss"])
    )

    # ----------------------------------------------
    # LOSS CHART
    # ----------------------------------------------

    st.subheader("Biểu đồ Train Loss và Validation Loss")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        history.history["loss"],
        label="Train Loss",
        color="blue"
    )

    ax.plot(
        history.history["val_loss"],
        label="Validation Loss",
        color="orange"
    )

    ax.set_title("Loss Train vs Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    show_chart(fig)

    # ----------------------------------------------
    # DỰ BÁO TRÊN TẬP TEST
    # ----------------------------------------------

    st.header("4. KIỂM THỬ MÔ HÌNH")

    test_pred = model.predict(
        X_test,
        verbose=0
    )

    test_pred = scaler.inverse_transform(test_pred)

    y_test_real = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    )

    # ----------------------------------------------
    # ĐÁNH GIÁ
    # ----------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_test_real,
            test_pred
        )
    )

    mae = mean_absolute_error(
        y_test_real,
        test_pred
    )

    r2 = r2_score(
        y_test_real,
        test_pred
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "RMSE",
        f"{rmse:.4f}"
    )

    c2.metric(
        "MAE",
        f"{mae:.4f}"
    )

    c3.metric(
        "R²",
        f"{r2:.4f}"
    )

    st.caption(
        "RMSE và MAE càng thấp càng tốt. "
        "R² càng gần 1 càng tốt; R² có thể âm."
    )

    # ----------------------------------------------
    # BIỂU ĐỒ TEST
    # ----------------------------------------------

    st.subheader("So sánh giá thực tế và dự báo")

    fig, ax = plt.subplots(figsize=(12, 6))ax.plot(
        y_test_real.flatten(),
        label="Giá thật",
        color="blue"
    )

    ax.plot(
        test_pred.flatten(),
        label="Giá dự báo",
        color="orange"
    )

    ax.set_title("Giá thực tế vs Giá dự báo LSTM")
    ax.set_xlabel("Phiên giao dịch")
    ax.set_ylabel("Giá")
    ax.legend()
    ax.grid(alpha=0.3)

    show_chart(fig)

    # ----------------------------------------------
    # DỰ BÁO TƯƠNG LAI
    # ----------------------------------------------

    st.header("5. DỰ BÁO GIÁ CỔ PHIẾU TƯƠNG LAI")

    # Giữ nguyên cách xử lý của code gốc
    last_60_days = np.vstack((
        train_scaled,
        test_scaled
    ))[-60:]

    last_60_days = last_60_days.reshape(
        1, 60, 1
    )

    future_predictions = []

    with st.spinner("Đang dự báo các phiên tiếp theo..."):

        for i in range(future_days):

            next_pred = model.predict(
                last_60_days,
                verbose=0
            )

            future_predictions.append(
                next_pred[0, 0]
            )

            last_60_days = np.append(
                last_60_days[:, 1:, :],
                next_pred.reshape(1, 1, 1),
                axis=1
            )

    future_predictions = np.array(
        future_predictions
    ).reshape(-1, 1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    )

    # ----------------------------------------------
    # TẠO BẢNG DỰ BÁO
    # ----------------------------------------------

    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=future_days
    )

    forecast_df = pd.DataFrame({
        "Ngày dự kiến": future_dates,
        "Giá dự báo": future_predictions.flatten()
    })

    forecast_df["Thay đổi (%)"] = (
        (
            forecast_df["Giá dự báo"] / last_price
        ) - 1
    ) * 100

    st.subheader(
        f"Bảng dự báo {future_days} phiên tiếp theo"
    )

    st.dataframe(
        forecast_df.style.format({
            "Giá dự báo": "{:,.2f}",
            "Thay đổi (%)": "{:+.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ----------------------------------------------
    # BIỂU ĐỒ TỔNG HỢP GIỐNG CODE GỐC
    # ----------------------------------------------

    st.subheader(
        "Biểu đồ giá thật, dự báo Test và tương lai"
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        y_test_real.flatten(),
        label="Giá thật",
        color="blue"
    )

    ax.plot(
        test_pred.flatten(),
        label="Dự báo Test",
        color="orange"
    )

    future_x = range(
        len(y_test_real),
        len(y_test_real) + future_days
    )

    ax.plot(
        future_x,future_predictions.flatten(),
        marker="o",
        linestyle="--",
        color="red",
        label=f"{future_days} phiên tới"
    )

    ax.set_title(
        f"Dự báo giá cổ phiếu {ticker} bằng LSTM"
    )

    ax.set_xlabel("Phiên giao dịch")
    ax.set_ylabel("Giá")
    ax.legend()
    ax.grid(alpha=0.3)

    show_chart(fig)

    # ----------------------------------------------
    # BIỂU ĐỒ RIÊNG TƯƠNG LAI
    # ----------------------------------------------

    st.subheader("Xu hướng giá dự báo tương lai")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        forecast_df["Ngày dự kiến"],
        forecast_df["Giá dự báo"],
        marker="o",
        color="red",
        linewidth=2
    )

    ax.set_title(
        f"Dự báo {future_days} phiên giao dịch tới"
    )

    ax.set_xlabel("Ngày dự kiến")
    ax.set_ylabel("Giá dự báo")
    ax.grid(alpha=0.3)

    fig.autofmt_xdate()
    show_chart(fig)

    # ----------------------------------------------
    # NHẬN XÉT
    # ----------------------------------------------

    st.header("6. PHÂN TÍCH KẾT QUẢ")

    final_price = float(future_predictions[-1, 0])

    change_percent = (
        (final_price - last_price)
        / last_price
    ) * 100

    a, b, c = st.columns(3)

    a.metric(
        "Giá hiện tại",
        f"{last_price:,.2f}"
    )

    b.metric(
        f"Giá sau {future_days} phiên",
        f"{final_price:,.2f}"
    )

    c.metric(
        "Biến động dự báo",
        f"{change_percent:+.2f}%"
    )

    if change_percent > 0:
        st.success(
            f"Mô hình LSTM dự báo cổ phiếu {ticker} "
            f"có xu hướng TĂNG khoảng "
            f"{change_percent:.2f}% trong "
            f"{future_days} phiên tiếp theo."
        )

    elif change_percent < 0:
        st.warning(
            f"Mô hình LSTM dự báo cổ phiếu {ticker} "
            f"có xu hướng GIẢM khoảng "
            f"{abs(change_percent):.2f}% trong "
            f"{future_days} phiên tiếp theo."
        )

    else:
        st.info(
            "Mô hình dự báo giá tương đối ổn định."
        )

    if r2 < 0:
        st.warning(
            "R² âm: mô hình có kết quả kiểm thử "
            "kém hơn dự báo hằng số bằng trung bình "
            "giá của tập kiểm thử."
        )

    # ----------------------------------------------
    # XUẤT CSV
    # ----------------------------------------------

    st.header("7. TẢI KẾT QUẢ")

    csv = forecast_df.to_csv(
        index=False,
        encoding="utf-8-sig"
    )

    st.download_button(
        label="📥 TẢI KẾT QUẢ DỰ BÁO CSV",
        data=csv,
        file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
        mime="text/csv",use_container_width=True
    )

    st.info(
        "Ngày dự kiến chỉ loại trừ thứ Bảy và Chủ nhật, "
        "chưa loại trừ ngày nghỉ lễ của thị trường."
    )

    st.warning(
        "Kết quả chỉ phục vụ học tập và nghiên cứu. "
        "Dự báo bằng LSTM có thể sai lệch đáng kể "
        "so với giá cổ phiếu thực tế và không phải "
        "khuyến nghị đầu tư."
    )

else:

    st.info(
        "👈 Hãy chọn mã cổ phiếu và nhấn "
        "'BẮT ĐẦU DỰ BÁO' để chạy mô hình."
    )

    st.markdown("""
    ### Quy trình dự báo

    1. Tải dữ liệu cổ phiếu từ Yahoo Finance.
    2. Lấy dữ liệu giá đóng cửa (Close).
    3. Chia dữ liệu thành 80% Train và 20% Test.
    4. Chuẩn hóa dữ liệu bằng MinMaxScaler.
    5. Tạo chuỗi dữ liệu 60 phiên.
    6. Huấn luyện mô hình LSTM hai tầng.
    7. Đánh giá RMSE, MAE, R².
    8. Dự báo giá cổ phiếu trong tương lai.
    """)

st.divider()

st.caption(
    "Ứng dụng dự báo giá cổ phiếu | "
    "Streamlit + TensorFlow + LSTM + Yahoo Finance"
)
    # ----------------------------------------------
