// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace PassthroughCameraSamples.YoloBenchmark.Editor
{
    [CustomEditor(typeof(YoloBenchmarkRunner))]
    public sealed class YoloBenchmarkRunnerEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            var runner = (YoloBenchmarkRunner)target;
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Runtime status", runner.Status);

            using (new EditorGUI.DisabledScope(!Application.isPlaying || runner.IsRunning))
            {
                if (GUILayout.Button("Start Benchmark"))
                {
                    runner.StartBenchmark();
                }
            }

            using (new EditorGUI.DisabledScope(!Application.isPlaying || !runner.IsRunning))
            {
                if (GUILayout.Button("Stop Benchmark"))
                {
                    runner.StopBenchmark();
                }
            }

            using (new EditorGUI.DisabledScope(string.IsNullOrEmpty(runner.LastOutputDirectory)))
            {
                if (GUILayout.Button("Reveal Last Output"))
                {
                    EditorUtility.RevealInFinder(runner.LastOutputDirectory);
                }
            }
        }
    }

    public static class YoloBenchmarkSetupEditor
    {
        private const string RootFolder = "Assets/PassthroughCameraApiSamples/YoloBenchmark";
        private const string ConfigFolder = RootFolder + "/Config";
        private const string ConfigPath = ConfigFolder + "/YoloBenchmarkConfig.asset";
        private const string ScenePath = RootFolder + "/YoloBenchmark.unity";

        [MenuItem("Meta/YOLO Benchmark/Create Or Reset Benchmark Scene")]
        private static void CreateBenchmarkScene()
        {
            EnsureFolder(ConfigFolder);

            var config = AssetDatabase.LoadAssetAtPath<YoloBenchmarkConfig>(ConfigPath);
            if (config == null)
            {
                config = ScriptableObject.CreateInstance<YoloBenchmarkConfig>();
                config.InitializeOfficialNanoDefaults();
                AssetDatabase.CreateAsset(config, ConfigPath);
            }
            else if (config.Models == null || config.Models.Length == 0)
            {
                config.InitializeOfficialNanoDefaults();
                EditorUtility.SetDirty(config);
            }

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            var runnerObject = new GameObject("YoloBenchmarkRunner");
            var runner = runnerObject.AddComponent<YoloBenchmarkRunner>();
            var serializedRunner = new SerializedObject(runner);
            serializedRunner.FindProperty("m_config").objectReferenceValue = config;
            serializedRunner.FindProperty("m_runOnStart").boolValue = true;
            serializedRunner.ApplyModifiedPropertiesWithoutUndo();

            var cameraObject = new GameObject("BenchmarkCamera", typeof(Camera));
            cameraObject.tag = "MainCamera";
            var camera = cameraObject.GetComponent<Camera>();
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.black;
            camera.nearClipPlane = 0.1f;
            camera.farClipPlane = 10f;

            EditorSceneManager.MarkSceneDirty(scene);
            EditorSceneManager.SaveScene(scene, ScenePath);
            AddSceneToBuildSettings(ScenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Selection.activeGameObject = runnerObject;

            EditorUtility.DisplayDialog(
                "YOLO Benchmark",
                $"Created benchmark scene and config.\n\nAssign model assets in:\n{ConfigPath}",
                "OK");
        }

        [MenuItem("Meta/YOLO Benchmark/Select Benchmark Config")]
        private static void SelectBenchmarkConfig()
        {
            var config = AssetDatabase.LoadAssetAtPath<YoloBenchmarkConfig>(ConfigPath);
            if (config == null)
            {
                EditorUtility.DisplayDialog(
                    "YOLO Benchmark",
                    "Create the benchmark scene first.",
                    "OK");
                return;
            }

            Selection.activeObject = config;
            EditorGUIUtility.PingObject(config);
        }

        private static void EnsureFolder(string folderPath)
        {
            if (AssetDatabase.IsValidFolder(folderPath))
            {
                return;
            }

            var parent = Path.GetDirectoryName(folderPath)?.Replace('\\', '/');
            var folderName = Path.GetFileName(folderPath);
            if (!string.IsNullOrEmpty(parent) && !AssetDatabase.IsValidFolder(parent))
            {
                EnsureFolder(parent);
            }

            AssetDatabase.CreateFolder(parent, folderName);
        }

        private static void AddSceneToBuildSettings(string scenePath)
        {
            var scenes = new List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);
            for (var i = scenes.Count - 1; i >= 0; i--)
            {
                if (scenes[i].path == scenePath)
                {
                    scenes.RemoveAt(i);
                }
            }

            scenes.Insert(0, new EditorBuildSettingsScene(scenePath, true));
            EditorBuildSettings.scenes = scenes.ToArray();
        }
    }
}
