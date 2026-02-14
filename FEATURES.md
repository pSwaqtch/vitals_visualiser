# ADPD7000 Evaluation Tool — Feature Checklist

## Phase 1: Data Viewer
- [x] Load ADPD7000 xlsx exports (Data + metadata sheets)
- [x] Load generic CSV/Excel files
- [x] Auto-detect timestamp and signal columns
- [x] User-overridable column mapping (dropdowns)
- [x] Interactive Plotly chart (zoom, pan, hover, legend toggle)
- [x] Range slider for navigation
- [x] Metadata display (board, version, export time, ref BP)
- [x] Sample rate and duration calculation
- [x] Raw data table (expandable)
- [x] Sample data selector + file upload
- [x] Analysis window slider with box-select zoom
- [x] Export selected range as CSV download

## Phase 2: Data Input
- [x] Read data from xlsx exports
- [x] Read data from generic CSV files
- [x] Stream live data from WebSocket source

## Phase 3: HRV Analysis (via HeartPy)
- [x] Bandpass filtering for HRV (configurable cutoffs)
- [x] Peak detection visualization
- [x] HR (heart rate, BPM)
- [x] IBI (inter-beat interval)
- [x] RMSSD (root mean square of successive differences)
- [x] SDNN (standard deviation of NN intervals)
- [x] pNN50 / pNN20
- [x] SD1 / SD2 (Poincare plot descriptors)
- [x] SD1/SD2 ratio
- [x] Breathing rate estimation
- [x] RR tachogram
- [x] Poincare plot with SD1/SD2 ellipse

### HeartPy Features — Not Yet Integrated
- [ ] Clipping detection and interpolation (corrupted signal recovery)
- [ ] Peak enhancement (improve signal quality before analysis)
- [ ] Hampel filter / correction (outlier handling)
- [ ] Segmentwise analysis (split signal into segments, analyse each)
- [ ] Frequency-domain HRV (LF, HF, LF/HF ratio, VLF)
- [ ] SDSD (standard deviation of successive differences)
- [ ] HRV triangular index
- [ ] Sample rate estimation from data (`hp.get_samplerate_datetime`)
- [ ] Heart rate over time calculation (`calc_rr` / rolling HR)
- [ ] ECG signal processing mode
- [ ] Working data / measures dict export for downstream use

## Phase 4: Signal Processing
- [ ] Bandpass filtering as a general pre-processing step (not just HRV)
- [ ] Baseline wander removal
- [ ] Motion artifact detection
- [ ] Signal quality index (SQI)
- [ ] Multi-channel overlay / stacked subplots

## Phase 5: Vital Signs
- [ ] SpO2 estimation — ratio of ratios (Slot A / Slot B)
- [ ] Blood pressure estimation — PTT-based

## Phase 6: Multi-Modal
- [ ] ECG channel support
- [ ] EEG channel support
- [ ] BioZ channel support
- [ ] IMU channel support (accel/gyro)
- [ ] Cross-modal synchronization view

## Phase 7: Export & Reporting
- [x] Export windowed data as CSV
- [ ] Session summary report (PDF)
- [ ] Batch processing mode
