// Copyright (c) Meta Platforms, Inc. and affiliates.
using System.IO;
using Meta.XR.Samples;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection.Editor
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    [CustomEditor(typeof(SentisInferenceRunManager))]
    public class SentisModelEditorConverter : UnityEditor.Editor
    {
        private SentisInferenceRunManager m_targetClass;

        public void OnEnable()
        {
            m_targetClass = (SentisInferenceRunManager)target;
        }

        public override void OnInspectorGUI()
        {
            _ = DrawDefaultInspector();

            if (GUILayout.Button("Generate Sentis model from assigned ONNX"))
            {
                OnEnable(); // Get the latest values from the serialized object
                ConvertModel(); // convert the ONNX model to sentis
            }
        }

        private void ConvertModel()
        {
            if (m_targetClass.OnnxModel == null)
            {
                Debug.LogError("No ONNX model assigned.");
                return;
            }

            var model = ModelLoader.Load(m_targetClass.OnnxModel);
            var outputPath = Path.ChangeExtension(AssetDatabase.GetAssetPath(m_targetClass.OnnxModel), ".sentis");
            if (m_targetClass.SelectedYoloVersionUsesEndToEndOutput)
            {
                SaveQuantizedModel(model, outputPath);
                return;
            }

            //Here we transform the output of the model by feeding it through a Non-Max-Suppression layer.
            var graph = new FunctionalGraph();
            var input = graph.AddInput(model, 0);

            var centersToCornersData = new[]
            {
                        1,      0,      1,      0,
                        0,      1,      0,      1,
                        -0.5f,  0,      0.5f,   0,
                        0,      -0.5f,  0,      0.5f
            };
            var centersToCorners = Functional.Constant(new TensorShape(4, 4), centersToCornersData);
            var modelOutput = Functional.Forward(model, input)[0];
            if (m_targetClass.SelectedSingleOutputUsesChannelsLast)
            {
                var boxCoords = modelOutput[0, .., ..4];
                var allScores = modelOutput[0, .., 4..];
                var channelsLastScores = Functional.ReduceMax(allScores, 1);
                var channelsLastClassIDs = Functional.ArgMax(allScores, 1);
                var channelsLastCorners = Functional.MatMul(boxCoords, centersToCorners);
                var channelsLastModelFinal = graph.Compile(channelsLastCorners, channelsLastClassIDs, channelsLastScores);

                SaveQuantizedModel(channelsLastModelFinal, outputPath);
                return;
            }

            var channelsFirstBoxCoords = modelOutput[0, ..4, ..].Transpose(0, 1);
            var channelsFirstScores = modelOutput[0, 4.., ..].Transpose(0, 1);
            var scores = Functional.ReduceMax(channelsFirstScores, 1);
            var classIDs = Functional.ArgMax(channelsFirstScores, 1);
            var corners = Functional.MatMul(channelsFirstBoxCoords, centersToCorners);
            var modelFinal = graph.Compile(corners, classIDs, scores);

            SaveQuantizedModel(modelFinal, outputPath);
        }

        private static void SaveQuantizedModel(Model model, string outputPath)
        {
            ModelQuantizer.QuantizeWeights(QuantizationType.Uint8, ref model);
            ModelWriter.Save(outputPath, model);
            // refresh assets
            AssetDatabase.Refresh();
            Debug.Log($"Saved Sentis model to '{outputPath}'.");
        }
    }
}
