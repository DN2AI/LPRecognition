# LPRecognition
- Đây đơn giản chỉ là một dự án cho cuộc thi <b>Khoa Học Kĩ Thuật</b>.
- Với các thành viên của nhóm:
    + Võ Phương Nghi 11T1
    + Châu Nguyễn Thanh Duy 11L
## How It Works ?
- Nhận diện biển số xe theo các bước:
  + Xác định vị trí biển số xe dùng thuật toán <b>YOLOv7</b> (có thể dùng các phiên bản khác).
  + Phân đoạn kí tự (tách kí tự từ biển số xe).
  + Nhận diện kí tự đã được tách (OCR).

## Note
- (10/9/2023) Hoàn thành việc dùng model <b>YOLOv7</b> để phát hiện vị trí biển số xe. Đồng thời chỉnh sửa code để hiện background màu xanh.
![demo](doc/demo.jpg)

- (19/9/2023) Dùng thư viện <b>EasyOCR</b> để đọc biển số xe. Tuy nhiên lại không cho ra kết quả chính xác, làm tôi rất thất vọng.<br/>

- (20/9/2023) Từ việc dùng thư viện <b>EasyOCR</b> để đọc biển số xe không hiệu quả, tôi quyết định chuyển sang <b>PaddleOCR</b>. Và thật tuyệt vời, nó cho ra các kết quả cực kì chính xác! Điều này làm tôi rất bất ngờ, LPRecognition gần như đã hoàn thành.<br/>
![demo](https://github.com/DN2AI/LPRecognition/assets/55396370/8e3f9a7b-6038-4a5a-9c14-9aba1c1bbd31)

## Todo
- [X] Nhận diện biển số xe qua hình ảnh
- [X] Dùng OCR để đọc biển số xe.
- [X] Nhận diện biển số xe qua video.
- [ ] Nhận diện biển số xe real-time qua camera.
