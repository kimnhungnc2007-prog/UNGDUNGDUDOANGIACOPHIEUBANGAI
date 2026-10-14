        ax.plot(test_dates, y_test_real.flatten(), label="Giá thực tế")
        ax.plot(test_dates, test_pred.flatten(), label="Giá dự báo")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Loss trong quá trình huấn luyện")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history.history["loss"], label="Train Loss")
        ax.plot(history.history["val_loss"], label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "RMSE và MAE đo sai số dự báo, càng thấp thường càng tốt. "
            "R² có thể âm khi mô hình dự báo kém. Chỉ số trên tập kiểm tra không bảo đảm "
            "độ chính xác trong tương lai."
        )

    # ===================== DỰ BÁO TƯƠNG LAI =====================
    # Fit lại scaler và mô hình trên toàn bộ dữ liệu để dự báo tương lai.
    # Việc đánh giá ở tab trên vẫn dùng mô hình chỉ học trên tập train.
    full_scaler = MinMaxScaler()
    full_scaled = full_scaler.fit_transform(data)

    X_full, y_full = make_sequences(full_scaled, window)
    X_full = X_full.reshape(-1, window, 1)

    tf.keras.backend.clear_session()
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    future_model = build_model(window, units)
    future_model.fit(
        X_full,
        y_full,
        epochs=max(1, len(history.history["loss"])),
        batch_size=batch_size,
        shuffle=False,
        verbose=0
    )

    current_window = full_scaled[-window:].reshape(1, window, 1)
    future_scaled = []
    for _ in range(future_days):
        next_value = future_model.predict(current_window, verbose=0)[0, 0]
        future_scaled.append(next_value)
        current_window = np.concatenate(
            [current_window[:, 1:, :], np.array(next_value).reshape(1, 1, 1)],
            axis=1
        )

    future_prices = full_scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    future_dates = pd.bdate_range(
        start=pd.Timestamp(df.index[-1]) + pd.offsets.BDay(1),
        periods=future_days
    )
    forecast_df = pd.DataFrame({
        "Ngày dự báo": future_dates.strftime("%d/%m/%Y"),
        "Giá dự báo": future_prices
    })

    with tab3:
        st.subheader(f"Dự báo {future_days} phiên tiếp theo")
        st.caption("Ngày dự báo bỏ qua cuối tuần nhưng chưa loại trừ toàn bộ ngày nghỉ lễ.")
        st.dataframe(
            forecast_df.style.format({"Giá dự báo": "{:,.2f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Tải bảng dự báo CSV",
            data=forecast_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"du_bao_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )

        fig, ax = plt.subplots(figsize=(12, 5))
        recent = df.tail(120)
        ax.plot(recent.index, recent["Close"], label="Giá lịch sử")
        ax.plot(future_dates, future_prices, marker="o", linestyle="--", label="Giá dự báo")
        ax.set_title(f"Dự báo giá cổ phiếu {ticker}")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Giá")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.warning(
            "Dự báo nhiều phiên sử dụng các giá dự báo trước làm đầu vào, vì vậy sai số có thể tích lũy. "
            "Đây là kết quả mô hình học máy, không phải cam kết về giá thị trường."
        )

    with tab4:
        st.subheader("Dữ liệu lịch sử")
        st.dataframe(df.tail(200), use_container_width=True)
        st.download_button(
            "⬇️ Tải dữ liệu lịch sử CSV",
            data=df.to_csv(index=True).encode("utf-8-sig"),
            file_name=f"lich_su_{ticker.replace('.', '_')}.csv",
            mime="text/csv"
        )
