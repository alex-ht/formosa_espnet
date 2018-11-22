# README
## 簡介
這份流程是用來建置比賽系統，差別在
 * 比賽系統是ctc_weight=0.3
 * 比賽是完整的"train"當作訓練集, "test"當作驗證集
 * 比賽系統沒有用RNNLM
## Requirement
 * ESPnet: 可以使用ESPnet的[docker image](https://github.com/espnet/espnet/blob/master/docker/README.md)
 * cconv: 簡體中文轉台灣正體
 * uconv: Unicode normalization
 * dos2unix: 避免windows和linux文字檔格式問題
 * 支援zh_TW.UTF-8的Linux環境(或許可以用"C.UTF-8"，但沒測試)

參考指令

`apt-get install cconv libicu-dev language-pack-zh-hant`
