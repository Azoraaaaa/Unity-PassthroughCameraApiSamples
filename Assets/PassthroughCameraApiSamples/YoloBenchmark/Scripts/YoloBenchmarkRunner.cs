// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.Profiling;
using Stopwatch = System.Diagnostics.Stopwatch;

namespace PassthroughCameraSamples.YoloBenchmark
{
    public sealed class YoloBenchmarkRunner : MonoBehaviour
    {
        [SerializeField] private YoloBenchmarkConfig m_config;
        [SerializeField] private bool m_runOnStart = true;

        private readonly List<BenchmarkSample> m_samples = new List<BenchmarkSample>();
        private readonly List<BenchmarkRunSummary> m_runSummaries = new List<BenchmarkRunSummary>();
        private readonly List<double> m_currentRunFrameTimesMs = new List<double>();

        private bool m_isRunning;
        private bool m_stopRequested;
        private bool m_recordFrameTimes;
        private string m_status = "Idle";
        private string m_outputDirectory;

        public bool IsRunning => m_isRunning;
        public string Status => m_status;
        public string LastOutputDirectory => m_outputDirectory;

        private IEnumerator Start()
        {
            if (!m_runOnStart)
            {
                yield break;
            }

            yield return null;
            StartBenchmark();
        }

        private void Update()
        {
            if (m_recordFrameTimes)
            {
                m_currentRunFrameTimesMs.Add(Time.unscaledDeltaTime * 1000.0);
            }
        }

        private void OnDestroy()
        {
            m_stopRequested = true;
        }

        [ContextMenu("Start Benchmark")]
        public void StartBenchmark()
        {
            if (m_isRunning)
            {
                Debug.LogWarning($"{nameof(YoloBenchmarkRunner)} is already running.");
                return;
            }

            StartCoroutine(RunBenchmark());
        }

        [ContextMenu("Stop Benchmark")]
        public void StopBenchmark()
        {
            m_stopRequested = true;
        }

        private IEnumerator RunBenchmark()
        {
            if (m_config == null)
            {
                Debug.LogError($"{nameof(YoloBenchmarkRunner)} has no benchmark config assigned.");
                yield break;
            }

            if (m_config.Models == null || m_config.Models.Length == 0)
            {
                Debug.LogError($"{nameof(YoloBenchmarkRunner)} config has no model definitions.");
                yield break;
            }

            if (m_config.Backends == null || m_config.Backends.Length == 0)
            {
                Debug.LogError($"{nameof(YoloBenchmarkRunner)} config has no backends.");
                yield break;
            }

            m_isRunning = true;
            m_stopRequested = false;
            m_samples.Clear();
            m_runSummaries.Clear();

            var previousTargetFrameRate = Application.targetFrameRate;
            Application.targetFrameRate = m_config.TargetRenderRate;

            var sessionId = $"{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N").Substring(0, 8)}";
            m_outputDirectory = Path.Combine(Application.persistentDataPath, m_config.OutputFolderName, sessionId);
            Directory.CreateDirectory(m_outputDirectory);
            var device = YoloBenchmarkReportUtility.CaptureDeviceMetadata();

            Debug.Log(
                $"[YoloBenchmark] SESSION_START id={sessionId} protocol={m_config.ProtocolVersion} " +
                $"output='{m_outputDirectory}'");

            foreach (var modelDefinition in m_config.Models)
            {
                if (m_stopRequested)
                {
                    break;
                }

                if (modelDefinition == null)
                {
                    Debug.LogWarning("[YoloBenchmark] Skipping a null model definition.");
                    continue;
                }

                foreach (var backend in m_config.Backends)
                {
                    if (m_stopRequested)
                    {
                        break;
                    }

                    if (m_config.CollectGarbageBetweenRuns)
                    {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                        GC.Collect();
                    }

                    yield return RunConfiguration(sessionId, device, modelDefinition, backend);

                    if (!m_stopRequested && m_config.CooldownSeconds > 0f)
                    {
                        m_status = $"Cooling down for {m_config.CooldownSeconds:0.#} seconds";
                        yield return new WaitForSecondsRealtime(m_config.CooldownSeconds);
                    }
                }
            }

            try
            {
                YoloBenchmarkReportUtility.WriteReports(
                    m_outputDirectory,
                    m_config.ProtocolVersion,
                    sessionId,
                    device,
                    m_samples,
                    m_runSummaries);
            }
            catch (Exception exception)
            {
                Debug.LogError($"[YoloBenchmark] Failed to write reports: {exception}");
            }

            Application.targetFrameRate = previousTargetFrameRate;
            m_isRunning = false;
            m_status = m_stopRequested ? "Stopped; partial report written" : "Completed";

            Debug.Log(
                $"[YoloBenchmark] SESSION_END id={sessionId} status={m_status} " +
                $"samples={m_samples.Count} output='{m_outputDirectory}'");
        }

        private IEnumerator RunConfiguration(
            string sessionId,
            BenchmarkDeviceMetadata device,
            YoloBenchmarkModelDefinition modelDefinition,
            BackendType backend)
        {
            var runId = $"{sessionId}_{modelDefinition.Id}_{backend}";
            var summary = CreateBaseSummary(runId, modelDefinition, backend);
            m_status = $"Preparing {modelDefinition.Id} on {backend}";

            if (modelDefinition.Model == null)
            {
                summary.success = false;
                summary.error = "Model asset is not assigned.";
                m_runSummaries.Add(summary);
                Debug.LogWarning($"[YoloBenchmark] SKIP run={runId}: {summary.error}");
                yield break;
            }

            Model model;
            var modelLoadStart = Stopwatch.GetTimestamp();
            try
            {
                model = ModelLoader.Load(modelDefinition.Model);
            }
            catch (Exception exception)
            {
                model = null;
                summary.success = false;
                summary.error = $"Model loading failed: {exception.Message}";
            }

            if (model == null)
            {
                m_runSummaries.Add(summary);
                Debug.LogError($"[YoloBenchmark] FAIL run={runId}: {summary.error}");
                yield break;
            }
            summary.model_load_ms = TicksToMilliseconds(modelLoadStart, Stopwatch.GetTimestamp());

            if (!TryGetInputShape(model, out var inputSize, out var inputShape, out var inputError))
            {
                summary.success = false;
                summary.error = inputError;
                m_runSummaries.Add(summary);
                Debug.LogError($"[YoloBenchmark] FAIL run={runId}: {summary.error}");
                yield break;
            }

            summary.input_width = inputSize.x;
            summary.input_height = inputSize.y;
            summary.output_count = model.outputs.Count;

            Worker worker = null;
            Tensor<float> input = null;
            RenderTexture generatedTexture = null;
            var workerCreationStart = Stopwatch.GetTimestamp();
            try
            {
                worker = new Worker(model, backend);
                input = new Tensor<float>(inputShape);
            }
            catch (Exception exception)
            {
                input?.Dispose();
                worker?.Dispose();
                input = null;
                worker = null;
                summary.success = false;
                summary.error = $"Worker creation failed: {exception.Message}";
            }

            if (worker == null || input == null)
            {
                m_runSummaries.Add(summary);
                Debug.LogError($"[YoloBenchmark] FAIL run={runId}: {summary.error}");
                yield break;
            }
            summary.worker_creation_ms = TicksToMilliseconds(workerCreationStart, Stopwatch.GetTimestamp());

            var sourceTexture = m_config.InputTextureOverride;
            if (sourceTexture == null)
            {
                generatedTexture = CreateDeterministicTexture(inputSize.x, inputSize.y);
                sourceTexture = generatedTexture;
            }

            var textureTransform = new TextureTransform().SetTensorLayout(TensorLayout.NCHW);
            Debug.Log(
                $"[YoloBenchmark] RUN_START run={runId} model={modelDefinition.Id} backend={backend} " +
                $"input={inputSize.x}x{inputSize.y} warmup={m_config.WarmupIterations} " +
                $"samples={m_config.MeasuredIterations} utc={DateTime.UtcNow:O}");

            var runFailed = false;
            var warmupValues = new List<double>(m_config.WarmupIterations);
            for (var i = 0; i < m_config.WarmupIterations && !m_stopRequested; i++)
            {
                m_status = $"Warm-up {modelDefinition.Id} {backend}: {i + 1}/{m_config.WarmupIterations}";
                var warmupResult = new BenchmarkIterationResult();
                yield return ExecuteIteration(worker, input, sourceTexture, textureTransform, model.outputs.Count, warmupResult);
                if (!string.IsNullOrEmpty(warmupResult.error))
                {
                    summary.error = $"Warm-up failed: {warmupResult.error}";
                    runFailed = true;
                    break;
                }

                warmupValues.Add(warmupResult.totalMs);
                if (m_config.YieldOneFrameBetweenIterations)
                {
                    yield return null;
                }
            }

            var latencyValues = new List<double>(m_config.MeasuredIterations);
            var preprocessingValues = new List<double>(m_config.MeasuredIterations);
            var schedulingValues = new List<double>(m_config.MeasuredIterations);
            var executionValues = new List<double>(m_config.MeasuredIterations);
            var memoryStart = Profiler.GetTotalAllocatedMemoryLong();
            var memoryPeak = memoryStart;
            var batteryStart = SystemInfo.batteryLevel;
            var measuredRunStart = Stopwatch.GetTimestamp();
            m_currentRunFrameTimesMs.Clear();
            m_recordFrameTimes = !runFailed;

            for (var i = 0; i < m_config.MeasuredIterations && !m_stopRequested && !runFailed; i++)
            {
                m_status = $"Measuring {modelDefinition.Id} {backend}: {i + 1}/{m_config.MeasuredIterations}";
                var result = new BenchmarkIterationResult();
                yield return ExecuteIteration(worker, input, sourceTexture, textureTransform, model.outputs.Count, result);
                if (!string.IsNullOrEmpty(result.error))
                {
                    summary.error = $"Measurement failed at iteration {i}: {result.error}";
                    runFailed = true;
                    break;
                }

                summary.output_shape = result.outputShape;
                preprocessingValues.Add(result.preprocessingMs);
                schedulingValues.Add(result.schedulingMs);
                executionValues.Add(result.executionReadbackMs);
                latencyValues.Add(result.totalMs);

                var allocatedMemory = Profiler.GetTotalAllocatedMemoryLong();
                memoryPeak = Math.Max(memoryPeak, allocatedMemory);
                m_samples.Add(new BenchmarkSample
                {
                    protocol_version = m_config.ProtocolVersion,
                    session_id = sessionId,
                    run_id = runId,
                    timestamp_utc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                    device_model = device.device_model,
                    operating_system = device.operating_system,
                    unity_version = device.unity_version,
                    model_id = modelDefinition.Id,
                    model_version = modelDefinition.VersionLabel,
                    model_variant = modelDefinition.Variant,
                    model_asset = modelDefinition.Model.name,
                    backend = backend.ToString(),
                    quantization = modelDefinition.Quantization.ToString(),
                    output_kind = modelDefinition.OutputKind.ToString(),
                    tensor_layout = modelDefinition.TensorLayout.ToString(),
                    parameters_millions = modelDefinition.ParametersMillions,
                    gflops_at_640 = modelDefinition.GflopsAt640,
                    input_width = inputSize.x,
                    input_height = inputSize.y,
                    output_count = model.outputs.Count,
                    output_shape = result.outputShape,
                    iteration = i,
                    preprocessing_ms = result.preprocessingMs,
                    scheduling_ms = result.schedulingMs,
                    execution_readback_ms = result.executionReadbackMs,
                    total_ms = result.totalMs,
                    completion_frame_ms = Time.unscaledDeltaTime * 1000.0,
                    allocated_memory_bytes = allocatedMemory,
                    reserved_memory_bytes = Profiler.GetTotalReservedMemoryLong(),
                    mono_used_memory_bytes = Profiler.GetMonoUsedSizeLong(),
                    battery_level = SystemInfo.batteryLevel,
                    battery_status = SystemInfo.batteryStatus.ToString(),
                    gc_gen0_count = GC.CollectionCount(0),
                    gc_gen1_count = GC.CollectionCount(1),
                    gc_gen2_count = GC.CollectionCount(2)
                });

                if (m_config.YieldOneFrameBetweenIterations)
                {
                    yield return null;
                }
            }

            m_recordFrameTimes = false;
            var measuredRunEnd = Stopwatch.GetTimestamp();
            summary.success = !runFailed && !m_stopRequested && latencyValues.Count == m_config.MeasuredIterations;
            if (m_stopRequested && string.IsNullOrEmpty(summary.error))
            {
                summary.error = "Benchmark stopped by user.";
            }

            summary.sample_count = latencyValues.Count;
            summary.warmup_ms = YoloBenchmarkReportUtility.CalculateStatistics(warmupValues);
            summary.first_warmup_ms = warmupValues.Count > 0 ? warmupValues[0] : 0.0;
            summary.cold_start_total_ms = summary.model_load_ms + summary.worker_creation_ms + summary.first_warmup_ms;
            summary.preprocessing_ms = YoloBenchmarkReportUtility.CalculateStatistics(preprocessingValues);
            summary.scheduling_ms = YoloBenchmarkReportUtility.CalculateStatistics(schedulingValues);
            summary.execution_readback_ms = YoloBenchmarkReportUtility.CalculateStatistics(executionValues);
            summary.total_ms = YoloBenchmarkReportUtility.CalculateStatistics(latencyValues);
            summary.frame_ms = YoloBenchmarkReportUtility.CalculateStatistics(m_currentRunFrameTimesMs);
            summary.mean_inference_fps = summary.total_ms.mean > 0.0 ? 1000.0 / summary.total_ms.mean : 0.0;

            var measuredSeconds = TicksToMilliseconds(measuredRunStart, measuredRunEnd) / 1000.0;
            summary.sustained_samples_per_second = measuredSeconds > 0.0 ? summary.sample_count / measuredSeconds : 0.0;
            summary.dropped_frame_count = YoloBenchmarkReportUtility.CountSlowFrames(
                m_currentRunFrameTimesMs,
                m_config.TargetRenderRate);
            summary.memory_start_bytes = memoryStart;
            summary.memory_end_bytes = Profiler.GetTotalAllocatedMemoryLong();
            summary.memory_peak_bytes = memoryPeak;
            summary.battery_start = batteryStart;
            summary.battery_end = SystemInfo.batteryLevel;
            m_runSummaries.Add(summary);

            DisposeRunResources(worker, input, generatedTexture, model.outputs.Count);

            Debug.Log(
                $"[YoloBenchmark] RUN_END run={runId} success={summary.success} " +
                $"samples={summary.sample_count} mean_ms={summary.total_ms.mean:0.###} " +
                $"p95_ms={summary.total_ms.p95:0.###} sustained_fps={summary.sustained_samples_per_second:0.###} " +
                $"utc={DateTime.UtcNow:O}");
        }

        private static IEnumerator ExecuteIteration(
            Worker worker,
            Tensor<float> input,
            Texture sourceTexture,
            TextureTransform textureTransform,
            int outputCount,
            BenchmarkIterationResult result)
        {
            var totalStart = Stopwatch.GetTimestamp();

            var preprocessingStart = Stopwatch.GetTimestamp();
            try
            {
                TextureConverter.ToTensor(sourceTexture, input, textureTransform);
            }
            catch (Exception exception)
            {
                result.error = $"Texture preprocessing failed: {exception.Message}";
            }

            if (!string.IsNullOrEmpty(result.error))
            {
                yield break;
            }
            var preprocessingEnd = Stopwatch.GetTimestamp();

            var schedulingStart = Stopwatch.GetTimestamp();
            try
            {
                worker.Schedule(input);
            }
            catch (Exception exception)
            {
                result.error = $"Worker scheduling failed: {exception.Message}";
            }

            if (!string.IsNullOrEmpty(result.error))
            {
                yield break;
            }
            var schedulingEnd = Stopwatch.GetTimestamp();

            var executionStart = Stopwatch.GetTimestamp();
            var outputTensor = worker.PeekOutput(0) as Tensor<float>;
            if (outputTensor == null)
            {
                result.error = "Output 0 is not a float tensor.";
                yield break;
            }

            var outputAwaiter = outputTensor.ReadbackAndCloneAsync().GetAwaiter();
            while (!outputAwaiter.IsCompleted)
            {
                yield return null;
            }

            Tensor<float> output = null;
            try
            {
                output = outputAwaiter.GetResult();
                result.outputShape = output.shape.ToString();
                for (var i = 1; i < outputCount; i++)
                {
                    worker.PeekOutput(i)?.CompleteAllPendingOperations();
                }
            }
            catch (Exception exception)
            {
                result.error = $"Output synchronization failed: {exception.Message}";
            }
            finally
            {
                output?.Dispose();
            }

            if (!string.IsNullOrEmpty(result.error))
            {
                yield break;
            }

            var executionEnd = Stopwatch.GetTimestamp();
            result.preprocessingMs = TicksToMilliseconds(preprocessingStart, preprocessingEnd);
            result.schedulingMs = TicksToMilliseconds(schedulingStart, schedulingEnd);
            result.executionReadbackMs = TicksToMilliseconds(executionStart, executionEnd);
            result.totalMs = TicksToMilliseconds(totalStart, executionEnd);
        }

        private static bool TryGetInputShape(Model model, out Vector2Int inputSize, out TensorShape inputShape, out string error)
        {
            inputSize = default;
            inputShape = default;
            error = null;

            if (model.inputs.Count == 0)
            {
                error = "Model has no inputs.";
                return false;
            }

            var dynamicShape = model.inputs[0].shape;
            if (dynamicShape.isRankDynamic || dynamicShape.rank != 4)
            {
                error = $"Expected a static rank-4 NCHW input, got {dynamicShape}.";
                return false;
            }

            var batch = dynamicShape.Get(0);
            var channels = dynamicShape.Get(1);
            var height = dynamicShape.Get(2);
            var width = dynamicShape.Get(3);
            if (batch != 1 || channels != 3 || height <= 0 || width <= 0)
            {
                error = $"Expected input [1, 3, height, width], got {dynamicShape}.";
                return false;
            }

            inputSize = new Vector2Int(width, height);
            inputShape = new TensorShape(batch, channels, height, width);
            return true;
        }

        private static void DisposeRunResources(Worker worker, Tensor<float> input, RenderTexture generatedTexture, int outputCount)
        {
            if (worker != null)
            {
                try
                {
                    for (var i = 0; i < outputCount; i++)
                    {
                        worker.PeekOutput(i)?.CompleteAllPendingOperations();
                    }
                }
                catch (Exception exception)
                {
                    Debug.LogWarning($"[YoloBenchmark] Failed to complete pending output during cleanup: {exception.Message}");
                }
            }

            input?.Dispose();
            worker?.Dispose();
            if (generatedTexture != null)
            {
                generatedTexture.Release();
                Destroy(generatedTexture);
            }
        }

        private static RenderTexture CreateDeterministicTexture(int width, int height)
        {
            var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, true)
            {
                name = $"YoloBenchmarkSource_{width}x{height}"
            };
            var pixels = new Color32[width * height];
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var index = y * width + x;
                    pixels[index] = new Color32(
                        (byte)((x * 13 + y * 3) & 0xff),
                        (byte)((x * 5 + y * 11) & 0xff),
                        (byte)((x * 7 + y * 17) & 0xff),
                        255);
                }
            }

            texture.SetPixels32(pixels);
            texture.Apply(false, false);

            var renderTexture = new RenderTexture(width, height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear)
            {
                name = $"YoloBenchmarkInput_{width}x{height}"
            };
            renderTexture.Create();
            Graphics.Blit(texture, renderTexture);
            Destroy(texture);
            return renderTexture;
        }

        private BenchmarkRunSummary CreateBaseSummary(
            string runId,
            YoloBenchmarkModelDefinition definition,
            BackendType backend)
        {
            return new BenchmarkRunSummary
            {
                run_id = runId,
                model_id = definition.Id,
                model_version = definition.VersionLabel,
                model_variant = definition.Variant,
                model_asset = definition.Model != null ? definition.Model.name : string.Empty,
                backend = backend.ToString(),
                quantization = definition.Quantization.ToString(),
                output_kind = definition.OutputKind.ToString(),
                tensor_layout = definition.TensorLayout.ToString(),
                parameters_millions = definition.ParametersMillions,
                gflops_at_640 = definition.GflopsAt640,
                notes = definition.Notes,
                success = false,
                warmup_ms = new BenchmarkStatistics(),
                preprocessing_ms = new BenchmarkStatistics(),
                scheduling_ms = new BenchmarkStatistics(),
                execution_readback_ms = new BenchmarkStatistics(),
                total_ms = new BenchmarkStatistics(),
                frame_ms = new BenchmarkStatistics()
            };
        }

        private static double TicksToMilliseconds(long start, long end)
        {
            return (end - start) * 1000.0 / Stopwatch.Frequency;
        }

        private sealed class BenchmarkIterationResult
        {
            public double preprocessingMs;
            public double schedulingMs;
            public double executionReadbackMs;
            public double totalMs;
            public string outputShape;
            public string error;
        }
    }
}
