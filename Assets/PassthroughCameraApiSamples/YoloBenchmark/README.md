# Quest 3 YOLO Benchmark

This module provides a standalone, repeatable benchmark for comparing the
smallest official detection model from each supported YOLO generation on
Quest 3.

It intentionally excludes passthrough capture, depth raycasts, spatial
anchors, bounding-box drawing, and application UI. Those systems should be
measured separately as a full MR pipeline experiment.

## Official Model Matrix

Use COCO-pretrained detection models with batch size 1 and a fixed NCHW input.

| Benchmark ID | Official model | Parameters | FLOPs at 640 | Default output |
|---|---|---:|---:|---|
| `yolov9t` | YOLOv9t | 2.0 M | 7.7 B | Raw, channels-first |
| `yolov10n` | YOLOv10n | 2.3 M | 6.7 B | End-to-end |
| `yolo11n` | YOLO11n | 2.6 M | 6.5 B | Raw, channels-first |
| `yolo12n` | YOLO12n | 2.6 M | 6.5 B | Raw, channels-first |
| `yolo26n` | YOLO26n | 2.4 M | 5.4 B | End-to-end |

The Params/FLOPs metadata comes from the official Ultralytics model pages:

- <https://docs.ultralytics.com/models/yolov9/>
- <https://docs.ultralytics.com/models/yolov10/>
- <https://docs.ultralytics.com/models/yolo11/>
- <https://docs.ultralytics.com/models/yolo12/>
- <https://docs.ultralytics.com/models/yolo26/>

## Create The Benchmark Scene

1. Open the Unity project.
2. Run `Meta > YOLO Benchmark > Create Or Reset Benchmark Scene`.
3. Select the generated config:
   `Assets/PassthroughCameraApiSamples/YoloBenchmark/Config/YoloBenchmarkConfig.asset`.
4. Assign the imported `.onnx` or `.sentis` asset to each model entry.
5. Set the correct quantization metadata for every assigned asset.
6. Build and deploy to Quest 3.

The menu places `YoloBenchmark.unity` first in Build Settings so the benchmark
starts directly after launch.

Duplicate a model entry when comparing multiple deployment variants, for
example:

```text
yolo11n_fp32
yolo11n_uint8
yolo11n_320
yolo11n_480
yolo11n_640
```

## Default Protocol

The generated config uses:

```text
Target render rate: 72 Hz
Backends: CPU, GPUCompute
Warm-up iterations: 10
Measured iterations: 100
Cooldown between runs: 10 seconds
Batch size: 1
Input: deterministic GPU texture
```

For final FYP results:

1. Charge the headset above 80 percent.
2. Disconnect the charging cable before measurement.
3. Record Quest model and Horizon OS version.
4. Keep room temperature and headset starting temperature consistent.
5. Reboot before the first test group.
6. Run every configuration at least three times.
7. Rotate or randomize model order between repeated sessions.
8. Use 640 as the official-model baseline.
9. Treat 320 and 480 as separate input-resolution experiments.
10. Keep the same model format, quantization, backend, and output-head policy
    within each comparison table.

Do not combine FP32 and Uint8 runs in the same architecture-only comparison.

## Metrics

`samples.csv` stores every measured iteration:

- Texture preprocessing time.
- Worker scheduling time.
- Model execution, synchronization, and output readback time.
- Total sample latency.
- Frame time at completion.
- Allocated, reserved, and Mono memory.
- Battery level and status.
- Garbage collection counters.
- Model metadata, backend, input size, output shape, and device metadata.

`summary.json` stores:

- Model loading and Worker creation time.
- First warm-up and total cold-start time.
- Mean, standard deviation, min, max, P50, P90, P95, and P99.
- Mean latency-derived inference FPS.
- Sustained samples per second.
- Frame-time statistics and slow-frame count.
- Start, end, and peak memory.
- Start and end battery level.
- Failed or skipped configuration details.

`scheduling_ms` is the duration of `Worker.Schedule`. Depending on backend,
some execution can occur during scheduling. `execution_readback_ms` includes
remaining execution, synchronization, and output transfer. Use `total_ms` as
the primary end-to-end model-pipeline latency.

The benchmark does not perform NMS or annotation rendering. Measure those in
the MultiObjectDetection scene as a separate application-pipeline experiment.

## Quest System Metrics

Run OVR Metrics Tool in Report Mode during each benchmark session. Its report
provides system-level values that Unity cannot reliably obtain directly:

- Application frame rate.
- CPU and GPU levels.
- CPU and GPU utilization.
- Thermal level and throttling.
- Stale frames, tears, and dropped frames.

The app prints `SESSION_START`, `RUN_START`, `RUN_END`, and `SESSION_END` UTC
markers so OVR Metrics or Perfetto data can be aligned with CSV rows.

Useful hzdb preparation commands:

```bash
npx -y @meta-quest/hzdb device health-check
npx -y @meta-quest/hzdb device configure-testing setup
npx -y @meta-quest/hzdb app launch --cold-start --wait-for-idle com.samples.passthroughcamera
npx -y @meta-quest/hzdb log -f
```

Optional Perfetto capture:

```bash
npx -y @meta-quest/hzdb perf start --mode vr --app com.samples.passthroughcamera -o yolo_benchmark
# Run the benchmark, then use the PID printed by the start command.
npx -y @meta-quest/hzdb perf stop <PID> yolo_benchmark
```

Restore testing settings afterward:

```bash
npx -y @meta-quest/hzdb device configure-testing restore
```

## Result Files

On Quest, Unity writes under:

```text
/storage/emulated/0/Android/data/com.samples.passthroughcamera/files/YoloBenchmark/<session-id>/
```

Each session contains:

```text
samples.csv
summary.json
```

List and download results through hzdb:

```bash
npx -y @meta-quest/hzdb files ls /storage/emulated/0/Android/data/com.samples.passthroughcamera/files/YoloBenchmark
npx -y @meta-quest/hzdb files pull /storage/emulated/0/Android/data/com.samples.passthroughcamera/files/YoloBenchmark/<session-id> ./BenchmarkResults
```

## Recommended FYP Analysis

Primary independent variables:

- YOLO version.
- Backend.
- Input resolution.
- Quantization.
- Output type.

Primary dependent variables:

- Mean and P95 `total_ms`.
- Sustained samples per second.
- Mean and P95 frame time.
- Slow-frame rate.
- Peak memory.
- Thermal level and throttling state.
- Battery drain per minute for long-run tests.

Recommended figures:

- Mean latency with P95 error bars by model and backend.
- Latency distribution box plots.
- FPS versus model FLOPs.
- Peak memory versus parameter count.
- Latency and thermal level over time.
- Accuracy versus Quest latency Pareto plot, using offline COCO/custom-dataset
  accuracy results.
