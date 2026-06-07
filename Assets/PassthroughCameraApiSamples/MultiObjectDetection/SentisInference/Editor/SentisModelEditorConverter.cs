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
            var modelName = m_targetClass.OnnxModel.name.ToLowerInvariant();
            var outputPath = Path.ChangeExtension(AssetDatabase.GetAssetPath(m_targetClass.OnnxModel), ".sentis");
            var isEndToEndModel = modelName.Contains("ete") || modelName.Contains("end2end");
            if (isEndToEndModel)
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
            var modelOutput = Functional.Forward(model, input)[0];  //shape(1,N,85)
            // Following for yolo model. in (1, 84, N) out put shape
            var boxCoords = modelOutput[0, ..4, ..].Transpose(0, 1);
            var allScores = modelOutput[0, 4.., ..].Transpose(0, 1);
            var scores = Functional.ReduceMax(allScores, 1);                                    //shape=(N)
            var classIDs = Functional.ArgMax(allScores, 1);                                     //shape=(N)
            var corners = Functional.MatMul(boxCoords, centersToCorners);                    //shape=(N,4)
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
