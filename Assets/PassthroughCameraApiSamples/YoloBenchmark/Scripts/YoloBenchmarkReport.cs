// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;

namespace PassthroughCameraSamples.YoloBenchmark
{
    internal sealed class BenchmarkSample
    {
        public string protocol_version;
        public string session_id;
        public string run_id;
        public string timestamp_utc;
        public string device_model;
        public string operating_system;
        public string unity_version;
        public string model_id;
        public string model_version;
        public string model_variant;
        public string model_asset;
        public string backend;
        public string quantization;
        public string output_kind;
        public string tensor_layout;
        public double parameters_millions;
        public double gflops_at_640;
        public int input_width;
        public int input_height;
        public int output_count;
        public string output_shape;
        public int iteration;
        public double preprocessing_ms;
        public double scheduling_ms;
        public double execution_readback_ms;
        public double total_ms;
        public double completion_frame_ms;
        public long allocated_memory_bytes;
        public long reserved_memory_bytes;
        public long mono_used_memory_bytes;
        public double battery_level;
        public string battery_status;
        public int gc_gen0_count;
        public int gc_gen1_count;
        public int gc_gen2_count;
    }

    [Serializable]
    internal sealed class BenchmarkReport
    {
        public string protocol_version;
        public string session_id;
        public string generated_utc;
        public string output_directory;
        public BenchmarkDeviceMetadata device;
        public BenchmarkRunSummary[] runs;
    }

    [Serializable]
    internal sealed class BenchmarkDeviceMetadata
    {
        public string device_model;
        public string device_name;
        public string operating_system;
        public string processor_type;
        public int processor_count;
        public int system_memory_mb;
        public string graphics_device_name;
        public string graphics_device_type;
        public string graphics_device_version;
        public int graphics_memory_mb;
        public string unity_version;
        public string application_version;
        public string application_identifier;
        public int target_frame_rate;
        public int vsync_count;
    }

    [Serializable]
    internal sealed class BenchmarkRunSummary
    {
        public string run_id;
        public bool success;
        public string error;
        public string model_id;
        public string model_version;
        public string model_variant;
        public string model_asset;
        public string backend;
        public string quantization;
        public string output_kind;
        public string tensor_layout;
        public string output_shape;
        public string notes;
        public float parameters_millions;
        public float gflops_at_640;
        public int input_width;
        public int input_height;
        public int output_count;
        public int sample_count;
        public double model_load_ms;
        public double worker_creation_ms;
        public double first_warmup_ms;
        public double cold_start_total_ms;
        public BenchmarkStatistics warmup_ms;
        public BenchmarkStatistics preprocessing_ms;
        public BenchmarkStatistics scheduling_ms;
        public BenchmarkStatistics execution_readback_ms;
        public BenchmarkStatistics total_ms;
        public BenchmarkStatistics frame_ms;
        public double mean_inference_fps;
        public double sustained_samples_per_second;
        public int dropped_frame_count;
        public long memory_start_bytes;
        public long memory_end_bytes;
        public long memory_peak_bytes;
        public float battery_start;
        public float battery_end;
    }

    [Serializable]
    internal sealed class BenchmarkStatistics
    {
        public int count;
        public double min;
        public double max;
        public double mean;
        public double standard_deviation;
        public double p50;
        public double p90;
        public double p95;
        public double p99;
    }

    internal static class YoloBenchmarkReportUtility
    {
        public static BenchmarkDeviceMetadata CaptureDeviceMetadata()
        {
            return new BenchmarkDeviceMetadata
            {
                device_model = SystemInfo.deviceModel,
                device_name = SystemInfo.deviceName,
                operating_system = SystemInfo.operatingSystem,
                processor_type = SystemInfo.processorType,
                processor_count = SystemInfo.processorCount,
                system_memory_mb = SystemInfo.systemMemorySize,
                graphics_device_name = SystemInfo.graphicsDeviceName,
                graphics_device_type = SystemInfo.graphicsDeviceType.ToString(),
                graphics_device_version = SystemInfo.graphicsDeviceVersion,
                graphics_memory_mb = SystemInfo.graphicsMemorySize,
                unity_version = Application.unityVersion,
                application_version = Application.version,
                application_identifier = Application.identifier,
                target_frame_rate = Application.targetFrameRate,
                vsync_count = QualitySettings.vSyncCount
            };
        }

        public static BenchmarkStatistics CalculateStatistics(List<double> values)
        {
            if (values == null || values.Count == 0)
            {
                return new BenchmarkStatistics();
            }

            var sorted = new List<double>(values);
            sorted.Sort();

            var sum = 0.0;
            for (var i = 0; i < sorted.Count; i++)
            {
                sum += sorted[i];
            }

            var mean = sum / sorted.Count;
            var varianceSum = 0.0;
            for (var i = 0; i < sorted.Count; i++)
            {
                var difference = sorted[i] - mean;
                varianceSum += difference * difference;
            }

            return new BenchmarkStatistics
            {
                count = sorted.Count,
                min = sorted[0],
                max = sorted[sorted.Count - 1],
                mean = mean,
                standard_deviation = Math.Sqrt(varianceSum / sorted.Count),
                p50 = Percentile(sorted, 0.50),
                p90 = Percentile(sorted, 0.90),
                p95 = Percentile(sorted, 0.95),
                p99 = Percentile(sorted, 0.99)
            };
        }

        public static int CountSlowFrames(List<double> frameTimesMs, int targetFrameRate)
        {
            if (frameTimesMs == null || frameTimesMs.Count == 0 || targetFrameRate <= 0)
            {
                return 0;
            }

            var thresholdMs = 1000.0 / targetFrameRate * 1.5;
            var count = 0;
            for (var i = 0; i < frameTimesMs.Count; i++)
            {
                if (frameTimesMs[i] > thresholdMs)
                {
                    count++;
                }
            }

            return count;
        }

        public static void WriteReports(
            string outputDirectory,
            string protocolVersion,
            string sessionId,
            BenchmarkDeviceMetadata device,
            List<BenchmarkSample> samples,
            List<BenchmarkRunSummary> summaries)
        {
            WriteSamplesCsv(Path.Combine(outputDirectory, "samples.csv"), samples);
            var report = new BenchmarkReport
            {
                protocol_version = protocolVersion,
                session_id = sessionId,
                generated_utc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                output_directory = outputDirectory,
                device = device,
                runs = summaries.ToArray()
            };
            File.WriteAllText(
                Path.Combine(outputDirectory, "summary.json"),
                JsonUtility.ToJson(report, true),
                Encoding.UTF8);
        }

        private static double Percentile(List<double> sortedValues, double percentile)
        {
            if (sortedValues.Count == 1)
            {
                return sortedValues[0];
            }

            var position = (sortedValues.Count - 1) * percentile;
            var lower = (int)Math.Floor(position);
            var upper = (int)Math.Ceiling(position);
            if (lower == upper)
            {
                return sortedValues[lower];
            }

            var fraction = position - lower;
            return sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * fraction;
        }

        private static void WriteSamplesCsv(string path, List<BenchmarkSample> samples)
        {
            using var writer = new StreamWriter(path, false, new UTF8Encoding(false));
            writer.WriteLine(
                "protocol_version,session_id,run_id,timestamp_utc,device_model,operating_system,unity_version," +
                "model_id,model_version,model_variant,model_asset,backend,quantization,output_kind,tensor_layout," +
                "parameters_millions,gflops_at_640,input_width,input_height,output_count,output_shape,iteration," +
                "preprocessing_ms,scheduling_ms,execution_readback_ms,total_ms,completion_frame_ms," +
                "allocated_memory_bytes,reserved_memory_bytes,mono_used_memory_bytes,battery_level,battery_status," +
                "gc_gen0_count,gc_gen1_count,gc_gen2_count");

            foreach (var sample in samples)
            {
                writer.WriteLine(string.Join(",", new[]
                {
                    EscapeCsv(sample.protocol_version),
                    EscapeCsv(sample.session_id),
                    EscapeCsv(sample.run_id),
                    EscapeCsv(sample.timestamp_utc),
                    EscapeCsv(sample.device_model),
                    EscapeCsv(sample.operating_system),
                    EscapeCsv(sample.unity_version),
                    EscapeCsv(sample.model_id),
                    EscapeCsv(sample.model_version),
                    EscapeCsv(sample.model_variant),
                    EscapeCsv(sample.model_asset),
                    EscapeCsv(sample.backend),
                    EscapeCsv(sample.quantization),
                    EscapeCsv(sample.output_kind),
                    EscapeCsv(sample.tensor_layout),
                    Format(sample.parameters_millions),
                    Format(sample.gflops_at_640),
                    sample.input_width.ToString(CultureInfo.InvariantCulture),
                    sample.input_height.ToString(CultureInfo.InvariantCulture),
                    sample.output_count.ToString(CultureInfo.InvariantCulture),
                    EscapeCsv(sample.output_shape),
                    sample.iteration.ToString(CultureInfo.InvariantCulture),
                    Format(sample.preprocessing_ms),
                    Format(sample.scheduling_ms),
                    Format(sample.execution_readback_ms),
                    Format(sample.total_ms),
                    Format(sample.completion_frame_ms),
                    sample.allocated_memory_bytes.ToString(CultureInfo.InvariantCulture),
                    sample.reserved_memory_bytes.ToString(CultureInfo.InvariantCulture),
                    sample.mono_used_memory_bytes.ToString(CultureInfo.InvariantCulture),
                    Format(sample.battery_level),
                    EscapeCsv(sample.battery_status),
                    sample.gc_gen0_count.ToString(CultureInfo.InvariantCulture),
                    sample.gc_gen1_count.ToString(CultureInfo.InvariantCulture),
                    sample.gc_gen2_count.ToString(CultureInfo.InvariantCulture)
                }));
            }
        }

        private static string EscapeCsv(string value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return string.Empty;
            }

            if (!value.Contains(",") && !value.Contains("\"") && !value.Contains("\n") && !value.Contains("\r"))
            {
                return value;
            }

            return $"\"{value.Replace("\"", "\"\"")}\"";
        }

        private static string Format(double value)
        {
            return value.ToString("0.######", CultureInfo.InvariantCulture);
        }
    }
}
