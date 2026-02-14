# ADPD7000 Evaluation Tool — Feature Checklist

## Phase 1: Data Viewer (current)
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
- [x] Y-axis range scaling (manual min/max)
- [x] Export selected range as CSV download
- [x] HRV analysis via HeartPy (HR, IBI, RMSSD, SDNN, pNN50, pNN20, SD1/SD2, breathing rate)
- [x] Bandpass filtering for HRV (configurable cutoffs)
- [x] Peak detection visualization

## Phase 2: Signal Processing
- [ ] Bandpass filtering (configurable cutoffs)
- [ ] Baseline wander removal
- [ ] Motion artifact detection
- [ ] Signal quality index (SQI)
- [ ] Multi-channel overlay / stacked subplots

## Phase 3: Vital Signs
- [ ] Heart rate (HR) — peak detection on PPG
- [ ] Heart rate variability (HRV) — time/frequency domain
- [ ] Respiratory rate (RR) — from PPG modulation
- [ ] SpO2 estimation — ratio of ratios (Slot A / Slot B)
- [ ] Blood pressure estimation — PTT-based

## Phase 4: Multi-Modal
- [ ] ECG channel support
- [ ] EEG channel support
- [ ] BioZ channel support
- [ ] IMU channel support (accel/gyro)
- [ ] Cross-modal synchronization view

## Phase 5: Export & Reporting
- [ ] Export processed signals to CSV
- [ ] Session summary report (PDF)
- [ ] Batch processing mode
