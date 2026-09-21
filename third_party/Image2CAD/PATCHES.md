# Image2CAD 移植记录（OpenCV 3.4 → OpenCV 5 / NumPy 2 / Python 3.14）

原仓库：https://github.com/adityaintwala/Image2CAD （Apache-2.0）
用途：假山/园林等**不规则线稿**的矢量化（对照 / 辅助）。

## 背景
原项目写于 2016~2020，依赖 OpenCV 3.4.2、旧版 Tesseract。为在本机
（Python 3.14 + OpenCV 5.0 + NumPy 2.5 + Tesseract 5.5）运行，做了以下**机械性移植**
（不改算法逻辑）。

## 改动清单（7 个文件，约 11 处）

### 1. `Image2CAD/Image2CAD.py`（headless + 中文路径）
- 禁用 `cv2.imshow` / `cv2.waitKey`（原代码会弹出 5 个窗口并阻塞）
- 给 `cv2.imread` / `cv2.imwrite` 打补丁：用 `np.fromfile + cv2.imdecode` 支持**中文路径**
  （否则读到 None，OpenCV 在 Windows 不支持非 ASCII 路径）

### 2. `Core/Features/Texts/pytesseract.py`（Tesseract 适配）
- `tesseract_cmd` → 本机路径（可用环境变量 `TESSERACT_CMD` 覆盖）
- `tessdata` 指向纯英文目录（`C:\tessdata`，避免中文路径）
- `run_tesseract`：Python 3 下 `stderr` 是 bytes，需 `decode('utf-8')`
- `image_to_string` 里 `-psm 6` → `--psm 6`（Tesseract 5 参数格式）

### 3. `Core/Features/Texts/TextsFeature.py`
- 同上：`config="-psm 6"` → `config="--psm 6"`

### 4~5. `findContours` 返回值变更（OpenCV 3 → 4/5）
- `Core/Features/Circles/CirclesFeature.py`：`ret, contours, hierarchy =` → `contours, hierarchy =`
- `Core/Features/Arrowheads/ArrowHeadsFeature.py`：`_im, contour, hierarchy =` → `contour, hierarchy =`
- （`TextsFeature.py` 用 `[-2]` 取值，天然兼容，无需改）

### 6. `Core/Features/LineSegments/LineSegmentsFeature.py`
- `HoughLinesP` 返回格式变化：3.x 是 (N,1,4)，5.0 是 (N,4) → 改为 `x1,y1,x2,y2 = np.asarray(line).reshape(-1)[:4]`
- `if None != lines.any():` → `if lines is not None:`（OpenCV 无结果时返回 None）
- `np.int0(corners)` → `np.asarray(corners).astype(int)`（NumPy 2 移除 `np.int0`）

### 7. `Core/Features/LineSegments/SpecialLineSegments.py` / `Core/Features/Cognition/Cognition.py`
- `np.int` → `int`（2 处）、`np.uint` → `int`（1 处）、`np.int0` → `int`（1 处）
  （NumPy 1.24+ 移除了这些旧别名）

## 运行方式
```powershell
# 在 Image2CAD/Image2CAD 目录下
..\..\..\.venv\Scripts\python.exe Image2CAD.py <图片路径>
```
输出：图片同目录的 `Output/<图名>/<时间戳>/` 内 —— `*.dxf` + `*.I2C` + 5 张中间 PNG。

## 验证
- ✅ 仓库自带 `TestData/1.png` 全流程跑通，导出 `1.dxf`（47.8 KB）
- ✅ Tesseract 5.5 正常调用（eng + osd）

## 备注
- 原项目**没有 ML 模型文件**，README 提到的 ML 部分在开源版里被剥离；OCR 依赖 Tesseract。
- 输出 DXF 与中间图可用于对照；本项目主流程（多Agent）不受影响。
