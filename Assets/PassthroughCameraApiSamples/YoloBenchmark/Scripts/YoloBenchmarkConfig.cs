// Copyright (c) Meta Platforms, Inc. and affiliates.

using System;
using Unity.InferenceEngine;
using UnityEngine;

namespace PassthroughCameraSamples.YoloBenchmark
{
    public enum YoloBenchmarkVersion
    {
        YoloV9 = 9,
        YoloV10 = 10,
        Yolo11 = 11,
        Yolo12 = 12,
        Yolo26 = 26
    }

    public enum YoloBenchmarkOutputKind
    {
        Raw,
        EndToEnd,
        PostProcessed
    }

    public enum YoloBenchmarkTensorLayout
    {
        ChannelsFirst,
        ChannelsLast
    }

    public enum YoloBenchmarkQuantization
    {
        Unspecified,
        Float32,
        Float16Weights,
        Uint8Weights
    }

    [Serializable]
    public sealed class YoloBenchmarkModelDefinition
    {
        [SerializeField] private string m_id;
        [SerializeField] private YoloBenchmarkVersion m_version;
        [SerializeField] private string m_variant;
        [SerializeField] private ModelAsset m_model;
        [SerializeField] private YoloBenchmarkOutputKind m_outputKind;
        [SerializeField] private YoloBenchmarkTensorLayout m_tensorLayout;
        [SerializeField] private YoloBenchmarkQuantization m_quantization;
        [SerializeField] private float m_parametersMillions;
        [SerializeField] private float m_gflopsAt640;
        [SerializeField] private string m_notes;

        public string Id => string.IsNullOrWhiteSpace(m_id) ? VersionLabel : m_id;
        public YoloBenchmarkVersion Version => m_version;
        public string Variant => m_variant;
        public ModelAsset Model => m_model;
        public YoloBenchmarkOutputKind OutputKind => m_outputKind;
        public YoloBenchmarkTensorLayout TensorLayout => m_tensorLayout;
        public YoloBenchmarkQuantization Quantization => m_quantization;
        public float ParametersMillions => m_parametersMillions;
        public float GflopsAt640 => m_gflopsAt640;
        public string Notes => m_notes;

        public string VersionLabel
        {
            get
            {
                switch (m_version)
                {
                    case YoloBenchmarkVersion.YoloV9:
                        return "YOLOv9";
                    case YoloBenchmarkVersion.YoloV10:
                        return "YOLOv10";
                    default:
                        return $"YOLO{(int)m_version}";
                }
            }
        }

        public YoloBenchmarkModelDefinition()
        {
        }

#if UNITY_EDITOR
        public YoloBenchmarkModelDefinition(
            string id,
            YoloBenchmarkVersion version,
            string variant,
            YoloBenchmarkOutputKind outputKind,
            YoloBenchmarkTensorLayout tensorLayout,
            float parametersMillions,
            float gflopsAt640,
            string notes)
        {
            m_id = id;
            m_version = version;
            m_variant = variant;
            m_outputKind = outputKind;
            m_tensorLayout = tensorLayout;
            m_quantization = YoloBenchmarkQuantization.Unspecified;
            m_parametersMillions = parametersMillions;
            m_gflopsAt640 = gflopsAt640;
            m_notes = notes;
        }
#endif
    }

    [CreateAssetMenu(fileName = "YoloBenchmarkConfig", menuName = "Meta/YOLO Benchmark Config")]
    public sealed class YoloBenchmarkConfig : ScriptableObject
    {
        [Header("Models")]
        [SerializeField] private YoloBenchmarkModelDefinition[] m_models = Array.Empty<YoloBenchmarkModelDefinition>();

        [Header("Execution")]
        [SerializeField] private BackendType[] m_backends = { BackendType.CPU, BackendType.GPUCompute };
        [SerializeField, Min(0)] private int m_warmupIterations = 10;
        [SerializeField, Min(1)] private int m_measuredIterations = 100;
        [SerializeField, Min(0f)] private float m_cooldownSeconds = 10f;
        [SerializeField] private bool m_yieldOneFrameBetweenIterations = true;
        [SerializeField] private bool m_collectGarbageBetweenRuns = true;

        [Header("Input")]
        [SerializeField, Tooltip("Leave empty to generate a deterministic GPU texture matching each model input size.")]
        private Texture m_inputTextureOverride;

        [Header("Reporting")]
        [SerializeField] private string m_protocolVersion = "quest3-yolo-nano-v1";
        [SerializeField] private string m_outputFolderName = "YoloBenchmark";
        [SerializeField, Min(1)] private int m_targetRenderRate = 72;

        public YoloBenchmarkModelDefinition[] Models => m_models;
        public BackendType[] Backends => m_backends;
        public int WarmupIterations => Math.Max(0, m_warmupIterations);
        public int MeasuredIterations => Math.Max(1, m_measuredIterations);
        public float CooldownSeconds => Math.Max(0f, m_cooldownSeconds);
        public bool YieldOneFrameBetweenIterations => m_yieldOneFrameBetweenIterations;
        public bool CollectGarbageBetweenRuns => m_collectGarbageBetweenRuns;
        public Texture InputTextureOverride => m_inputTextureOverride;
        public string ProtocolVersion => m_protocolVersion;
        public string OutputFolderName => string.IsNullOrWhiteSpace(m_outputFolderName) ? "YoloBenchmark" : m_outputFolderName;
        public int TargetRenderRate => Math.Max(1, m_targetRenderRate);

#if UNITY_EDITOR
        public void InitializeOfficialNanoDefaults()
        {
            m_models = new[]
            {
                new YoloBenchmarkModelDefinition(
                    "yolov9t",
                    YoloBenchmarkVersion.YoloV9,
                    "t",
                    YoloBenchmarkOutputKind.Raw,
                    YoloBenchmarkTensorLayout.ChannelsFirst,
                    2.0f,
                    7.7f,
                    "Official smallest YOLOv9 detection model."),
                new YoloBenchmarkModelDefinition(
                    "yolov10n",
                    YoloBenchmarkVersion.YoloV10,
                    "n",
                    YoloBenchmarkOutputKind.EndToEnd,
                    YoloBenchmarkTensorLayout.ChannelsLast,
                    2.3f,
                    6.7f,
                    "Official nano YOLOv10 detection model."),
                new YoloBenchmarkModelDefinition(
                    "yolo11n",
                    YoloBenchmarkVersion.Yolo11,
                    "n",
                    YoloBenchmarkOutputKind.Raw,
                    YoloBenchmarkTensorLayout.ChannelsFirst,
                    2.6f,
                    6.5f,
                    "Official nano YOLO11 detection model."),
                new YoloBenchmarkModelDefinition(
                    "yolo12n",
                    YoloBenchmarkVersion.Yolo12,
                    "n",
                    YoloBenchmarkOutputKind.Raw,
                    YoloBenchmarkTensorLayout.ChannelsFirst,
                    2.6f,
                    6.5f,
                    "Official nano YOLO12 detection model."),
                new YoloBenchmarkModelDefinition(
                    "yolo26n",
                    YoloBenchmarkVersion.Yolo26,
                    "n",
                    YoloBenchmarkOutputKind.EndToEnd,
                    YoloBenchmarkTensorLayout.ChannelsLast,
                    2.4f,
                    5.4f,
                    "Official nano YOLO26 detection model using the default one-to-one head.")
            };
        }
#endif
    }
}
