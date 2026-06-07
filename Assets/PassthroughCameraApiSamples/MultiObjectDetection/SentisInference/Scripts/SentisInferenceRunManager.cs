// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Meta.XR;
using Meta.XR.Samples;
using Unity.Collections;
using Unity.InferenceEngine;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisInferenceRunManager : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraAccess m_cameraAccess;
        [SerializeField] private DetectionUiMenuManager m_uiMenuManager;
        [SerializeField] private DetectionManager m_detectionManager;

        [Header("Sentis Model config")]
        [SerializeField] private BackendType m_backend = BackendType.CPU;
        [SerializeField] private ModelAsset m_sentisModel;
        [SerializeField] private TextAsset m_labelsAsset;
        [SerializeField, Range(0, 1)] private float m_iouThreshold = 0.6f;
        [SerializeField, Range(0, 1)] private float m_scoreThreshold = 0.23f;

        [Header("UI display references")]
        [SerializeField] private SentisInferenceUiManager m_uiInference;

        [Header("[Editor Only] Convert to Sentis")]
        public ModelAsset OnnxModel;
        [Space(40)]

        private Worker m_engine;
        private Vector2Int m_inputSize;
        private TensorShape m_inputTensorShape;
        private TextureTransform m_textureTransform;
        private int m_outputCount;
        private bool m_hasLoggedOutputShape;
        private bool m_hasLoggedDetectionCount;
        private bool m_hasLoggedAnchorWarning;
        private bool m_hasLoggedUnsupportedOutputShape;
        private readonly List<(int classId, Vector4 boundingBox)> m_detections = new List<(int classId, Vector4 boundingBox)>();

        private void Awake()
        {
            if (m_sentisModel == null)
            {
                Debug.LogError($"{nameof(SentisInferenceRunManager)} has no inference model assigned.");
                enabled = false;
                return;
            }

            var model = ModelLoader.Load(m_sentisModel);
            if (!TryGetModelInputShape(model, out m_inputSize, out m_inputTensorShape, out var inputShapeError))
            {
                Debug.LogError($"{nameof(SentisInferenceRunManager)} cannot use '{m_sentisModel.name}': {inputShapeError}");
                enabled = false;
                return;
            }

            m_textureTransform = new TextureTransform().SetTensorLayout(TensorLayout.NCHW);
            m_outputCount = model.outputs.Count;
            Debug.Log(
                $"{nameof(SentisInferenceRunManager)} loaded '{m_sentisModel.name}' " +
                $"with backend {m_backend}, input {m_inputTensorShape}, and {m_outputCount} output(s).");
            m_engine = new Worker(model, m_backend);
        }

        private IEnumerator Start()
        {
            m_uiInference.SetLabels(m_labelsAsset);

            while (true)
            {
                while (m_uiMenuManager.IsPaused)
                {
                    yield return null;
                }
                yield return RunInference();
            }
        }

        private void OnDestroy()
        {
            if (m_engine == null)
            {
                return;
            }

            CompletePendingOutputs(m_engine, m_outputCount);
            m_engine.Dispose();
        }

        internal static void PreloadModel(ModelAsset modelAsset)
        {
            if (modelAsset == null)
            {
                Debug.LogWarning($"{nameof(SentisInferenceRunManager)} cannot preload a null model asset.");
                return;
            }

            // Load model
            var model = ModelLoader.Load(modelAsset);
            if (!TryGetModelInputShape(model, out _, out var inputTensorShape, out var inputShapeError))
            {
                Debug.LogWarning($"{nameof(SentisInferenceRunManager)} cannot preload '{modelAsset.name}': {inputShapeError}");
                return;
            }

            // Create engine to run model
            using var worker = new Worker(model, BackendType.CPU);

            // Run inference with an empty image to load the model in the memory. The first inference blocks the main thread for a long time, so we're doing it on the app launch
            Texture tempTexture = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            var textureTransform = new TextureTransform().SetTensorLayout(TensorLayout.NCHW);
            using var input = new Tensor<float>(inputTensorShape);
            TextureConverter.ToTensor(tempTexture, input, textureTransform);
            worker.Schedule(input);

            // Complete the inference immediately and destroy the temporary texture
            CompletePendingOutputs(worker, model.outputs.Count);
            Destroy(tempTexture);
        }

        private IEnumerator RunInference()
        {
            if (!m_cameraAccess.IsPlaying)
            {
                yield break;
            }

            [DllImport("OVRPlugin", CallingConvention = CallingConvention.Cdecl)]
            static extern OVRPlugin.Result ovrp_GetNodePoseStateAtTime(double time, OVRPlugin.Node nodeId, out OVRPlugin.PoseStatef nodePoseState);
            if (!ovrp_GetNodePoseStateAtTime(OVRPlugin.GetTimeInSeconds(), OVRPlugin.Node.Head, out _).IsSuccess())
            {
                Debug.Log("ovrp_GetNodePoseStateAtTime failed, which means 'm_cameraAccess.GetCameraPose()' is not reliable, skipping.");
                yield break;
            }

            var cachedCameraPose = m_cameraAccess.GetCameraPose();

            // Update Capture data
            Texture targetTexture = m_cameraAccess.GetTexture();

            // Convert the texture to a Tensor and schedule the inference
            using var input = new Tensor<float>(m_inputTensorShape);
            TextureConverter.ToTensor(targetTexture, input, m_textureTransform);

            // Schedule all model layers
            m_engine.Schedule(input);

            if (m_outputCount == 1)
            {
                yield return ReadYoloEndToEndDetections();
            }
            else if (m_outputCount == 3)
            {
                yield return ReadPostProcessedDetections();
            }
            else
            {
                Debug.LogError($"Unsupported model output count: {m_outputCount}");
                yield break;
            }

            // Checking if spatial anchor is tracked ensures bounding boxes are placed at correct world space positIons.
            if (!m_cameraAccess.IsPlaying || m_detectionManager.m_spatialAnchor == null || !m_detectionManager.m_spatialAnchor.IsTracked)
            {
                if (!m_hasLoggedAnchorWarning)
                {
                    Debug.LogWarning(
                        $"Inference produced {m_detections.Count} detection(s), but annotations are skipped " +
                        "because the camera is not playing or the spatial anchor is not tracked.");
                    m_hasLoggedAnchorWarning = true;
                }
                yield break;
            }

            // Update UI.
            m_uiInference.DrawUIBoxes(m_detections, m_inputSize, cachedCameraPose);
        }

        private IEnumerator ReadYoloEndToEndDetections()
        {
            m_detections.Clear();

            var outputTensor = m_engine.PeekOutput(0) as Tensor<float>;
            if (outputTensor == null)
            {
                Debug.LogError("YOLO end-to-end output 0 is not a float tensor.");
                yield break;
            }

            var outputAwaiter = outputTensor.ReadbackAndCloneAsync().GetAwaiter();
            while (!outputAwaiter.IsCompleted)
            {
                yield return null;
            }

            using var output = outputAwaiter.GetResult();
            if (output.shape.length == 0)
            {
                yield break;
            }

            if (!m_hasLoggedOutputShape)
            {
                Debug.Log($"Single model output shape: {output.shape}");
                m_hasLoggedOutputShape = true;
            }

            if (!TryGetYoloEndToEndLayout(output.shape, out var detectionCount, out var channelsLast))
            {
                if (!m_hasLoggedUnsupportedOutputShape)
                {
                    Debug.LogError(
                        $"Unsupported single-output model shape: {output.shape}. " +
                        "YOLO26 end-to-end export must produce [1, detectionCount, 6]. " +
                        "Raw outputs such as [1, 84, 8400] need the editor converter instead.");
                    m_hasLoggedUnsupportedOutputShape = true;
                }
                yield break;
            }

            for (var i = 0; i < detectionCount; i++)
            {
                var score = channelsLast ? output[0, i, 4] : output[0, 4, i];
                if (score < m_scoreThreshold)
                {
                    continue;
                }

                var classId = Mathf.RoundToInt(channelsLast ? output[0, i, 5] : output[0, 5, i]);
                var rawBoundingBox = channelsLast
                    ? new Vector4(output[0, i, 0], output[0, i, 1], output[0, i, 2], output[0, i, 3])
                    : new Vector4(output[0, 0, i], output[0, 1, i], output[0, 2, i], output[0, 3, i]);
                if (!TryNormalizeAndClampBox(rawBoundingBox, m_inputSize, out var boundingBox))
                {
                    continue;
                }

                m_detections.Add((classId, boundingBox));
            }

            if (!m_hasLoggedDetectionCount)
            {
                Debug.Log($"YOLO end-to-end parser kept {m_detections.Count} detection(s) at score threshold {m_scoreThreshold}.");
                m_hasLoggedDetectionCount = true;
            }
        }

        private IEnumerator ReadPostProcessedDetections()
        {
            // Get the results. ReadbackAndCloneAsync waits for all layers to complete before returning the result
            var boxesTensor = m_engine.PeekOutput(0) as Tensor<float>;
            if (boxesTensor == null)
            {
                Debug.LogError("Post-processed output 0 is not a float boxes tensor.");
                m_detections.Clear();
                yield break;
            }

            var boxesAwaiter = boxesTensor.ReadbackAndCloneAsync().GetAwaiter();
            while (!boxesAwaiter.IsCompleted)
            {
                yield return null;
            }
            using var boxes = boxesAwaiter.GetResult();
            if (boxes.shape[0] == 0)
            {
                m_detections.Clear();
                yield break;
            }

            var classIDsTensor = m_engine.PeekOutput(1) as Tensor<int>;
            if (classIDsTensor == null)
            {
                Debug.LogError("Post-processed output 1 is not an int class ID tensor.");
                m_detections.Clear();
                yield break;
            }

            var classIDsAwaiter = classIDsTensor.ReadbackAndCloneAsync().GetAwaiter();
            while (!classIDsAwaiter.IsCompleted)
            {
                yield return null;
            }
            using var classIDs = classIDsAwaiter.GetResult();
            if (classIDs.shape[0] == 0)
            {
                Debug.LogError("classIDs.shape[0] == 0");
                m_detections.Clear();
                yield break;
            }

            var scoresTensor = m_engine.PeekOutput(2) as Tensor<float>;
            if (scoresTensor == null)
            {
                Debug.LogError("Post-processed output 2 is not a float scores tensor.");
                m_detections.Clear();
                yield break;
            }

            var scoresAwaiter = scoresTensor.ReadbackAndCloneAsync().GetAwaiter();
            while (!scoresAwaiter.IsCompleted)
            {
                yield return null;
            }
            using var scores = scoresAwaiter.GetResult();
            if (scores.shape[0] == 0)
            {
                Debug.LogError("scores.shape[0] == 0");
                m_detections.Clear();
                yield break;
            }

            NonMaxSuppression(m_detections, boxes, classIDs, scores, m_inputSize, m_iouThreshold, m_scoreThreshold);
        }

        private static void CompletePendingOutputs(Worker worker, int outputCount)
        {
            for (var i = 0; i < outputCount; i++)
            {
                worker.PeekOutput(i)?.CompleteAllPendingOperations();
            }
        }

        private static bool TryGetModelInputShape(Model model, out Vector2Int inputSize, out TensorShape inputTensorShape, out string error)
        {
            inputSize = default;
            inputTensorShape = default;
            error = null;

            if (model.inputs.Count == 0)
            {
                error = "model has no inputs.";
                return false;
            }

            var inputShape = model.inputs[0].shape;
            if (inputShape.isRankDynamic || inputShape.rank != 4)
            {
                error = $"expected a static rank-4 NCHW input, got {inputShape}.";
                return false;
            }

            var batch = inputShape.Get(0);
            var channels = inputShape.Get(1);
            var height = inputShape.Get(2);
            var width = inputShape.Get(3);
            if (batch <= 0 || channels != 3 || height <= 0 || width <= 0)
            {
                error = $"expected input [batch, 3, height, width], got {inputShape}.";
                return false;
            }

            inputSize = new Vector2Int(width, height);
            inputTensorShape = new TensorShape(batch, channels, height, width);
            return true;
        }

        private static bool TryGetYoloEndToEndLayout(TensorShape shape, out int detectionCount, out bool channelsLast)
        {
            detectionCount = 0;
            channelsLast = true;

            if (shape.rank != 3 || shape[0] != 1)
            {
                return false;
            }

            if (shape[2] == 6)
            {
                detectionCount = shape[1];
                channelsLast = true;
                return true;
            }

            if (shape[1] == 6)
            {
                detectionCount = shape[2];
                channelsLast = false;
                return true;
            }

            return false;
        }

        private static bool TryNormalizeAndClampBox(Vector4 box, Vector2 inputSize, out Vector4 normalizedBox)
        {
            normalizedBox = default;
            if (!IsFinite(box.x) || !IsFinite(box.y) || !IsFinite(box.z) || !IsFinite(box.w))
            {
                return false;
            }

            var maxCoordinate = Mathf.Max(Mathf.Abs(box.x), Mathf.Abs(box.y), Mathf.Abs(box.z), Mathf.Abs(box.w));
            if (maxCoordinate <= 1.5f)
            {
                box.x *= inputSize.x;
                box.z *= inputSize.x;
                box.y *= inputSize.y;
                box.w *= inputSize.y;
            }

            var left = Mathf.Clamp(Mathf.Min(box.x, box.z), 0f, inputSize.x);
            var top = Mathf.Clamp(Mathf.Min(box.y, box.w), 0f, inputSize.y);
            var right = Mathf.Clamp(Mathf.Max(box.x, box.z), 0f, inputSize.x);
            var bottom = Mathf.Clamp(Mathf.Max(box.y, box.w), 0f, inputSize.y);
            if (right - left <= Mathf.Epsilon || bottom - top <= Mathf.Epsilon)
            {
                return false;
            }

            normalizedBox = new Vector4(left, top, right, bottom);
            return true;
        }

        private static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

        private static void NonMaxSuppression(List<(int classId, Vector4 boundingBox)> outDetections, Tensor<float> boxes, Tensor<int> classIDs, Tensor<float> scores, Vector2 inputSize, float iouThreshold, float scoreThreshold)
        {
            outDetections.Clear();

            // Filter by score threshold first
            List<int> filteredIndices = new List<int>();
            NativeArray<float>.ReadOnly scoresArray = scores.AsReadOnlyNativeArray();
            for (int i = 0; i < scoresArray.Length; i++)
            {
                if (scoresArray[i] >= scoreThreshold)
                {
                    filteredIndices.Add(i);
                }
            }

            if (filteredIndices.Count == 0)
            {
                return;
            }

            // Sort filtered indices by scores in descending order
            filteredIndices.Sort((a, b) => scoresArray[b].CompareTo(scoresArray[a]));

            // Apply NMS algorithm
            bool[] suppressed = new bool[filteredIndices.Count];
            for (int i = 0; i < filteredIndices.Count; i++)
            {
                if (suppressed[i])
                    continue;

                int idx = filteredIndices[i];

                // Add this detection to results
                if (!TryNormalizeAndClampBox(GetBox(idx), inputSize, out var selectedBox))
                {
                    continue;
                }

                outDetections.Add((classIDs[idx], selectedBox));

                // Suppress overlapping boxes regardless of class
                for (int j = i + 1; j < filteredIndices.Count; j++)
                {
                    if (suppressed[j])
                        continue;

                    int jdx = filteredIndices[j];

                    if (TryNormalizeAndClampBox(GetBox(jdx), inputSize, out var candidateBox))
                    {
                        float iou = CalculateIoU(selectedBox, candidateBox);
                        if (iou > iouThreshold)
                        {
                            suppressed[j] = true;
                        }
                    }
                }
            }

            Vector4 GetBox(int i) => new Vector4(boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3]);
        }

        internal static float CalculateIoU(Vector4 boxA, Vector4 boxB)
        {
            // Boxes are in format (topLeftX, topLeftY, bottomRightX, bottomRightY)
            // Calculate intersection coordinates
            float x1 = Mathf.Max(boxA.x, boxB.x);
            float y1 = Mathf.Max(boxA.y, boxB.y);
            float x2 = Mathf.Min(boxA.z, boxB.z);
            float y2 = Mathf.Min(boxA.w, boxB.w);

            // Calculate intersection area
            float intersectionWidth = Mathf.Max(0, x2 - x1);
            float intersectionHeight = Mathf.Max(0, y2 - y1);
            float intersectionArea = intersectionWidth * intersectionHeight;

            // Calculate individual box areas
            float boxAArea = (boxA.z - boxA.x) * (boxA.w - boxA.y);
            float boxBArea = (boxB.z - boxB.x) * (boxB.w - boxB.y);

            // Calculate union area
            float unionArea = boxAArea + boxBArea - intersectionArea;

            // Return IoU (Intersection over Union)
            if (unionArea == 0)
                return 0;

            return intersectionArea / unionArea;
        }
    }
}
