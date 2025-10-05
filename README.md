# Stardist-in-bubble-segmentation
🗓 Tuần 1 (1/10 – 6/10):

Tìm hiểu bài toán segmentation và ứng dụng StarDist.

Đọc paper “Bubble identification from images with machine learning 
methods ” (Hessenkemper).

Cài đặt môi trường (Python, TensorFlow, StarDist, Jupyter).

Tạo repository GitHub và setup project structure.

🗓 Tuần 2 (7/10 – 13/10):

Thu thập yêu cầu hệ thống và xác định mục tiêu cụ thể.

Tìm hoặc chuẩn bị dataset.

Viết tài liệu mô tả yêu cầu (requirements.md).

🗓 Tuần 3 (14/10 – 20/10):

Thiết kế pipeline huấn luyện: chia train/test, augment data.

Xác định kiến trúc StarDist (2D hoặc 3D, số tia n_rays, patch size…).

Thiết kế kiến trúc hệ thống và UI (nếu có giao diện).

🗓 Tuần 4 (21/10 – 27/10):

Cài đặt và thử huấn luyện mô hình StarDist trên subset nhỏ của dataset.

Debug, kiểm tra input-output của mô hình.

Viết script huấn luyện train_stardist.py.

🗓 Tuần 5 (28/10 – 3/11):

Huấn luyện mô hình hoàn chỉnh với toàn bộ dataset.

Theo dõi loss, IoU, Dice coefficient.

Ghi lại kết quả vào file log và notebook.

Đánh giá sơ bộ kết quả dự đoán (visualize mask).

🗓 Tuần 6 (4/11 – 10/11):

Phân tích lỗi (error analysis).

Điều chỉnh hyperparameter (learning rate, batch size, n_rays).

So sánh kết quả sau khi tối ưu.

Viết tài liệu “Model Evaluation Report”.

🗓 Tuần 7 (11/11 – 17/11):

Bắt đầu phát triển ứng dụng web.

Thiết kế UI upload ảnh và hiển thị kết quả segmentation.

Kết nối mô hình StarDist với giao diện inference.

🗓 Tuần 8 (18/11 – 24/11):

Hoàn thiện giao diện: hiển thị ảnh gốc + ảnh mask.

Thử nghiệm pipeline end-to-end: upload → dự đoán → hiển thị.

Viết code tối ưu tốc độ dự đoán (GPU / batch inference).

🗓 Tuần 9 (25/11 – 1/12):

Kiểm thử toàn hệ thống (unit test + integration test).

Kiểm tra performance (FPS, latency).

Tối ưu code, làm sạch repo.

🗓 Tuần 10 (2/12 – 8/12):

Viết tài liệu hướng dẫn sử dụng (README.md, user manual).

Chuẩn bị slide báo cáo.

Ghi lại kết quả thực nghiệm, biểu đồ đánh giá.

🗓 Tuần 11 (9/12 – 15/12):

Rà soát toàn bộ project, hoàn thiện phần trình bày và code.

Ghi nhận đóng góp nhóm.

🗓 Tuần 12 (16/12 – 31/12):

Chuẩn bị demo, video trình chiếu kết quả.

Viết báo cáo tổng kết (report.pdf).

Nộp project và thuyết trình.
